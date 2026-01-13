import os
import gc
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusion3Pipeline

def load_image(path, size=512):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    return img


@torch.no_grad()
def encode_image_to_latents(pipe, image, device, dtype):
    arr = np.array(image).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
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


def _as_1d_float_timestep(t, device):
    if not torch.is_tensor(t):
        t = torch.tensor([t], device=device, dtype=torch.float32)
    else:
        t = t.to(device=device, dtype=torch.float32)
        if t.ndim == 0:
            t = t[None]
        elif t.ndim != 1:
            t = t.view(-1)
    return t


@torch.no_grad()
def get_text_embeddings(pipe, prompt, negative_prompt=""):
    device = pipe._execution_device
    return pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        prompt_3=None,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt,
        negative_prompt_3=None,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )


@torch.no_grad()
def sd3_velocity(pipe, latents, timestep, text_embeds_bundle, cfg_scale, low_vram_cfg=True):

    prompt_embeds, neg_prompt_embeds, pooled, neg_pooled = text_embeds_bundle

    t = _as_1d_float_timestep(timestep, latents.device)


    v_uncond = pipe.transformer(
        hidden_states=latents,
        timestep=t,
        encoder_hidden_states=neg_prompt_embeds,
        pooled_projections=neg_pooled,
        return_dict=False,
    )[0]

    v_cond = pipe.transformer(
        hidden_states=latents,
        timestep=t,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled,
        return_dict=False,
    )[0]

    return v_uncond + cfg_scale * (v_cond - v_uncond)



def _euler_update(sample, v, dt):
    return (sample.float() + dt * v.float()).to(sample.dtype)


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
    low_vram_cfg=True,
    empty_cache_every=8,
):
    device = pipe._execution_device
    x_src = x_src.to(device)
    g = torch.Generator(device=device).manual_seed(seed)

    pipe.scheduler.set_timesteps(T, device=device)
    timesteps = pipe.scheduler.timesteps
    sigmas = pipe.scheduler.sigmas.to(device)

    print("Scheduler:", type(pipe.scheduler).__name__)
    print("len(timesteps) =", len(timesteps))
    print("len(sigmas)    =", len(sigmas))

    emb_src = get_text_embeddings(pipe, prompt_src or "", negative_prompt="")
    emb_tar = get_text_embeddings(pipe, prompt_tar, negative_prompt="")

    z_fe = x_src.clone()

    for k, i in enumerate(range(nmax, nmin, -1), start=1):
        assert 0 <= i < len(timesteps)
        assert i + 1 < len(sigmas)

        t_i = _as_1d_float_timestep(timesteps[i], device)
        sigma_i = sigmas[i]

        v_delta_accum = None

        for _ in range(navg):
            n = randn_like_gen(x_src, g)

            # FlowMatch forward noising
            z_src_hat = pipe.scheduler.scale_noise(x_src, t_i, noise=n)
            z_tar_hat = z_fe + z_src_hat - x_src

            v_tar = sd3_velocity(pipe, z_tar_hat, t_i, emb_tar, cfg_tar, low_vram_cfg=low_vram_cfg)
            v_src = sd3_velocity(pipe, z_src_hat, t_i, emb_src, cfg_src, low_vram_cfg=low_vram_cfg)

            v_delta = (v_tar - v_src)
            v_delta_accum = v_delta if v_delta_accum is None else (v_delta_accum + v_delta)

        v_delta = v_delta_accum / float(navg)

        dt = (sigmas[i + 1] - sigmas[i]).item()
        z_fe = _euler_update(z_fe, v_delta, dt)

        if empty_cache_every and (k % empty_cache_every == 0) and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if nmin == 0:
        return z_fe

    # build z_tar at nmin
    t_m = _as_1d_float_timestep(timesteps[nmin], device)
    n = randn_like_gen(x_src, g)
    z_src_hat = pipe.scheduler.scale_noise(x_src, t_m, noise=n)
    z_tar = z_fe + z_src_hat - x_src

    #target sampling
    for i in range(nmin, 0, -1):
        t_i = _as_1d_float_timestep(timesteps[i], device)
        v = sd3_velocity(pipe, z_tar, t_i, emb_tar, cfg_tar, low_vram_cfg=low_vram_cfg)
        dt = (sigmas[i + 1] - sigmas[i]).item()
        z_tar = _euler_update(z_tar, v, dt)

    return z_tar


def randn_like_gen(x, g=None):
    if g is None:
        return torch.randn(x.shape, device=x.device, dtype=x.dtype)
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=g)


@torch.no_grad()
def sanity_one_step(pipe, x_src, emb_tar, cfg_tar, out_path="sanity_step.png", low_vram_cfg=True):
    device = pipe._execution_device
    x_src = x_src.to(device)

    pipe.scheduler.set_timesteps(10, device=device)
    timesteps = pipe.scheduler.timesteps
    sigmas = pipe.scheduler.sigmas.to(device)

    t0 = _as_1d_float_timestep(timesteps[0], device)
    z = x_src.clone()

    v = sd3_velocity(pipe, z, t0, emb_tar, cfg_tar, low_vram_cfg=low_vram_cfg)
    dt = (sigmas[1] - sigmas[0]).item()
    z2 = _euler_update(z, v, dt)

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
    nmin = int(os.environ.get("NMIN", "8"))
    navg = int(os.environ.get("NAVG", "4"))
    cfg_src = float(os.environ.get("CFG_SRC", "2"))
    cfg_tar = float(os.environ.get("CFG_TAR", "10"))
    seed = int(os.environ.get("SEED", "0"))

    drop_t5 = os.environ.get("DROP_T5", "1") == "1"
    low_vram_cfg = os.environ.get("LOW_VRAM_CFG", "1") == "1"

    print("-" * 60)
    print(f"seed={seed}  T={T}  nmax={nmax}  nmin={nmin}  navg={navg}  cfg_src={cfg_src}  cfg_tar={cfg_tar}")
    print(f"DROP_T5={drop_t5}  LOW_VRAM_CFG={low_vram_cfg}")
    print("-" * 60)

    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.float32

    print("Loading pipeline...")
    
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        text_encoder_3=None,
        tokenizer_3=None,
        torch_dtype=dtype if cuda else torch.float32,
        variant="fp16" if cuda else None,
        use_safetensors=True,
    )


    pipe.enable_sequential_cpu_offload()
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    exec_device = pipe._execution_device
    print("Execution device:", exec_device)

    img = load_image(test_img, size=512)
    x_src = encode_image_to_latents(pipe, img, device=exec_device, dtype=dtype)
    print("Encoded latents:", tuple(x_src.shape), x_src.dtype, x_src.device)

    emb_tar = get_text_embeddings(pipe, prompt_tar, negative_prompt="")
    sanity_one_step(pipe, x_src, emb_tar, cfg_tar, out_path="sanity_step.png", low_vram_cfg=low_vram_cfg)

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
        low_vram_cfg=low_vram_cfg,
    )

    out_img = decode_latents_to_image(pipe, z_out)
    out_img.save("flowedit_out.png")
    print("Saved: flowedit_out.png")

    # cleanup
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()