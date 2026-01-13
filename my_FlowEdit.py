import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusion3Pipeline


# ----------------------------
# IO
# ----------------------------
def load_image(path, size=512):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    return img


@torch.no_grad()
def encode_image_to_latents(pipe, image, device, dtype):
    arr = np.array(image).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    x = (x * 2.0 - 1.0).to(device=device, dtype=dtype)

    latents = pipe.vae.encode(x).latent_dist.sample()
    latents = latents * pipe.vae.config.scaling_factor
    return latents


@torch.no_grad()
def decode_latents_to_image(pipe, latents):
    latents = latents / pipe.vae.config.scaling_factor
    img = pipe.vae.decode(latents).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    img = img[0].permute(1, 2, 0).float().cpu().numpy()
    img = (img * 255).round().astype(np.uint8)
    return Image.fromarray(img)


# ----------------------------
# Text embeddings
# ----------------------------
@torch.no_grad()
def get_text_embeddings(pipe, prompt, negative_prompt=""):
    device = pipe._execution_device
    return pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        prompt_3=prompt,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt,
        negative_prompt_3=negative_prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )


# ----------------------------
# SD3 model output with CFG
# ----------------------------
@torch.no_grad()
def sd3_model_output(pipe, latents, timestep, text_embeds_bundle, cfg_scale):
    prompt_embeds, neg_prompt_embeds, pooled, neg_pooled = text_embeds_bundle

    latent_in = torch.cat([latents, latents], dim=0)
    noise_pred = pipe.transformer(
        hidden_states=latent_in,
        timestep=timestep,
        encoder_hidden_states=torch.cat([neg_prompt_embeds, prompt_embeds], dim=0),
        pooled_projections=torch.cat([neg_pooled, pooled], dim=0),
        return_dict=False,
    )[0]

    uncond, cond = noise_pred.chunk(2)
    return uncond + cfg_scale * (cond - uncond)


# ----------------------------
# Velocity in sigma-space from scheduler.step
# v ≈ (x_prev - x) / (sigma_prev - sigma)
# ----------------------------
@torch.no_grad()
def predict_velocity_sd3_sigma(pipe, latents, step_idx, timesteps, sigmas, text_embeds_bundle, cfg_scale):
    assert 0 <= step_idx < len(timesteps), "step_idx out of timesteps range"
    assert step_idx + 1 < len(sigmas), "sigmas should be at least len(timesteps)+1"

    t = timesteps[step_idx]
    sigma = float(sigmas[step_idx])
    sigma_prev = float(sigmas[step_idx + 1])

    model_out = sd3_model_output(pipe, latents, t, text_embeds_bundle, cfg_scale)
    out = pipe.scheduler.step(model_out, t, latents, return_dict=True)
    latents_prev = out.prev_sample

    d_sigma = sigma_prev - sigma
    if abs(d_sigma) < 1e-8:
        d_sigma = 1e-8

    v = (latents_prev - latents) / d_sigma
    return v


# ----------------------------
# FlowEdit (sigma-space, scheduler-consistent)
# ----------------------------
@torch.no_grad()
def flowedit(
    pipe,
    x_src,
    prompt_src,
    prompt_tar,
    T=50,
    nmax=33,
    nmin=0,
    navg=1,
    cfg_src=3.5,
    cfg_tar=13.5,
    seed=0,
):
    device = pipe._execution_device

    # Make sure x_src lives on execution device
    x_src = x_src.to(device)

    # RNG on correct device
    g = torch.Generator(device=device).manual_seed(seed)

    # Scheduler timeline
    pipe.scheduler.set_timesteps(T, device=device)
    timesteps = pipe.scheduler.timesteps
    sigmas = getattr(pipe.scheduler, "sigmas", None)
    if sigmas is None:
        raise RuntimeError("This scheduler has no `sigmas`. Cannot run sigma-space FlowEdit.")
    sigmas = sigmas.to(device)

    # Debug prints (keeps you from silently integrating nonsense)
    print("Scheduler:", type(pipe.scheduler).__name__)
    print("len(timesteps) =", len(timesteps))
    print("len(sigmas)    =", len(sigmas))
    print("timesteps head:", [int(t.item()) if t.numel() else t for t in timesteps[:3]])
    print("timesteps tail:", [int(t.item()) if t.numel() else t for t in timesteps[-3:]])
    print("sigmas head:", [float(s) for s in sigmas[:3]])
    print("sigmas tail:", [float(s) for s in sigmas[-3:]])

    # Embeddings
    emb_src = get_text_embeddings(pipe, prompt_src or "", negative_prompt="")
    emb_tar = get_text_embeddings(pipe, prompt_tar, negative_prompt="")

    # Init
    z_fe = x_src.clone()

    # Main FlowEdit loop: i = nmax ... nmin+1 (descending indices)
    # NOTE: diffusers timesteps[0] is typically the highest-noise step.
    for i in range(nmax, nmin, -1):
        assert i < len(timesteps), f"nmax/nmin indexing mismatch: i={i} >= len(timesteps)={len(timesteps)}"
        assert i + 1 < len(sigmas), f"sigmas too short for i={i}"

        v_delta_accum = 0.0

        for _ in range(navg):
            n = torch.randn_like(x_src, generator=g)

            # Use scheduler's add_noise (correct noising for this scheduler)
            z_src_hat = pipe.scheduler.add_noise(x_src, n, timesteps[i])
            z_tar_hat = z_fe + z_src_hat - x_src

            v_tar = predict_velocity_sd3_sigma(pipe, z_tar_hat, i, timesteps, sigmas, emb_tar, cfg_tar)
            v_src = predict_velocity_sd3_sigma(pipe, z_src_hat, i, timesteps, sigmas, emb_src, cfg_src)

            v_delta_accum = v_delta_accum + (v_tar - v_src)

        v_delta = v_delta_accum / float(navg)

        sigma = float(sigmas[i])
        sigma_prev = float(sigmas[i + 1])
        d_sigma = sigma_prev - sigma

        # Euler step in sigma-space
        z_fe = z_fe + d_sigma * v_delta

    if nmin == 0:
        return z_fe

    # If nmin > 0, finish with target sampling
    n = torch.randn_like(x_src, generator=g)
    z_src_hat = pipe.scheduler.add_noise(x_src, n, timesteps[nmin])
    z_tar = z_fe + z_src_hat - x_src

    for i in range(nmin, 0, -1):
        v = predict_velocity_sd3_sigma(pipe, z_tar, i, timesteps, sigmas, emb_tar, cfg_tar)
        d_sigma = float(sigmas[i + 1]) - float(sigmas[i])
        z_tar = z_tar + d_sigma * v

    return z_tar


# ----------------------------
# Small helper (avoid crashes on older diffusers)
# ----------------------------
def try_call(obj, name):
    fn = getattr(obj, name, None)
    if callable(fn):
        fn()
        print(f"[OK] {name}")
    else:
        print(f"[SKIP] {name} not available")


@torch.no_grad()
def sanity_one_step(pipe, x_src, emb_tar, cfg_tar, out_path="sanity_step.png"):
    """
    If this output is already glitchy, then your scheduler/model_output pairing is incompatible.
    """
    device = pipe._execution_device
    x_src = x_src.to(device)

    pipe.scheduler.set_timesteps(10, device=device)
    timesteps = pipe.scheduler.timesteps

    t0 = timesteps[0]
    z = x_src.clone()

    mo = sd3_model_output(pipe, z, t0, emb_tar, cfg_tar)
    z2 = pipe.scheduler.step(mo, t0, z).prev_sample

    img2 = decode_latents_to_image(pipe, z2)
    img2.save(out_path)
    print("Saved:", out_path)


def main():
    model_id = os.environ.get("SD3_MODEL_ID", "stabilityai/stable-diffusion-3-medium-diffusers")
    test_img = os.environ.get("TEST_IMG", "example_images/lighthouse.png")

    prompt_src = os.environ.get(
        "PROMPT_SRC",
        "The image features a tall white lighthouse standing prominently on a hill, with a beautiful blue sky in the background."
    )
    prompt_tar = os.environ.get(
        "PROMPT_TAR",
        "The image features Big Ben clock tower standing prominently on a hill, with a beautiful blue sky in the background."
    )

    T = int(os.environ.get("T", "50"))
    nmax = int(os.environ.get("NMAX", "33"))
    nmin = int(os.environ.get("NMIN", "0"))
    navg = int(os.environ.get("NAVG", "1"))
    cfg_src = float(os.environ.get("CFG_SRC", "3.5"))
    cfg_tar = float(os.environ.get("CFG_TAR", "13.5"))
    seed = int(os.environ.get("SEED", "0"))

    print("-" * 60)
    print(f"seed={seed}  T={T}  nmax={nmax}  nmin={nmin}  navg={navg}  cfg_src={cfg_src}  cfg_tar={cfg_tar}")
    print("-" * 60)

    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.float32

    print("Loading pipeline...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=dtype if cuda else torch.float32,
        variant="fp16" if cuda else None,
        use_safetensors=True,
    )

    # Memory mode
    try_call(pipe, "enable_model_cpu_offload")
    try_call(pipe, "enable_attention_slicing")

    exec_device = pipe._execution_device
    print("Execution device:", exec_device)

    # Load + encode
    img = load_image(test_img, size=512)
    x_src = encode_image_to_latents(pipe, img, device=exec_device, dtype=dtype)
    print("Encoded latents:", tuple(x_src.shape), x_src.dtype, x_src.device)

    # Sanity check (optional but very recommended)
    emb_tar = get_text_embeddings(pipe, prompt_tar, negative_prompt="")
    sanity_one_step(pipe, x_src, emb_tar, cfg_tar, out_path="sanity_step.png")

    # Run FlowEdit
    z_out = flowedit(
        pipe=pipe,
        x_src=x_src,
        prompt_src=prompt_src,
        prompt_tar=prompt_tar,
        T=T,
        nmax=nmax,
        nmin=nmin,
        navg=navg,
        cfg_src=cfg_src,
        cfg_tar=cfg_tar,
        seed=seed,
    )

    out_img = decode_latents_to_image(pipe, z_out)
    out_img.save("flowedit_out.png")
    print("Saved: flowedit_out.png")


if __name__ == "__main__":
    main()
