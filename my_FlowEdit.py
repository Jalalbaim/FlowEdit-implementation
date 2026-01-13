import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusion3Pipeline

def load_image(path, size=1024):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    return img

@torch.no_grad()
def encode_image_to_latents(pipe, image, device, dtype):
    # encode to vae
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    image = (image * 2.0 - 1.0).to(device=device, dtype=dtype)

    latents = pipe.vae.encode(image).latent_dist.sample()
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


@torch.no_grad()
def predict_velocity_sd3(pipe, latents, t_continuous, text_embeds_bundle, cfg_scale):
    """
    predict velocity V(x,t,c)
    """
    prompt_embeds, neg_prompt_embeds, pooled, neg_pooled = text_embeds_bundle

    t_int = int(round(float(t_continuous) * 1000))
    t_int = max(1, min(1000, t_int))
    timestep = torch.tensor([t_int], device=latents.device, dtype=torch.int64)

    # Classifier-free guidance: duplicate latents and embeddings
    latent_model_input = torch.cat([latents, latents], dim=0)

    # SD3 transformer forward
    noise_pred = pipe.transformer(
        hidden_states=latent_model_input,
        timestep=timestep,
        encoder_hidden_states=torch.cat([neg_prompt_embeds, prompt_embeds], dim=0),
        pooled_projections=torch.cat([neg_pooled, pooled], dim=0),
        return_dict=False,
    )[0]

    noise_uncond, noise_cond = noise_pred.chunk(2)
    noise_guided = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

    # V(x,t) = (eps_hat - x) / (1 - t)
    denom = max(1e-4, 1.0 - float(t_continuous))
    v = (noise_guided - latents) / denom
    return v

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
    device = x_src.device
    g = torch.Generator(device=device).manual_seed(seed)

    # time schedule: t_T=1 -> t_0=0
    ts = torch.linspace(1.0, 0.0, T + 1, device=device)
    ts = ts.clamp(1e-4, 1.0 - 1e-4)

    # embeddings
    emb_src = get_text_embeddings(pipe, prompt_src or "", negative_prompt="")
    emb_tar = get_text_embeddings(pipe, prompt_tar, negative_prompt="")

    # Init
    z_fe = x_src.clone()  # z_fe at t_nmax is x_src

    # i = nmax ... nmin+1
    for i in range(nmax, nmin, -1):
        t_i = float(ts[i].item())
        t_prev = float(ts[i - 1].item())
        dt = (t_prev - t_i) # < 0

        v_delta_accum = 0.0
        for _ in range(navg):
            # torch.randn_like ne supporte pas 'generator' sur certaines versions de PyTorch
            n = torch.randn(x_src.shape, generator=g, device=x_src.device, dtype=x_src.dtype)

            z_src_hat = (1.0 - t_i) * x_src + t_i * n
            z_tar_hat = z_fe + z_src_hat - x_src

            v_tar = predict_velocity_sd3(pipe, z_tar_hat, t_i, emb_tar, cfg_tar)
            v_src = predict_velocity_sd3(pipe, z_src_hat, t_i, emb_src, cfg_src)
            v_delta_accum = v_delta_accum + (v_tar - v_src)

        v_delta = v_delta_accum / float(navg)
        z_fe = z_fe + dt * v_delta  # Euler step

    if nmin == 0:
        return z_fe

    t_nmin = float(ts[nmin].item())
    # torch.randn_like ne supporte pas 'generator' sur certaines versions de PyTorch
    n = torch.randn(x_src.shape, generator=g, device=x_src.device, dtype=x_src.dtype)
    z_src_hat = (1.0 - t_nmin) * x_src + t_nmin * n
    z_tar = z_fe + z_src_hat - x_src

    for i in range(nmin, 0, -1):
        t_i = float(ts[i].item())
        t_prev = float(ts[i - 1].item())
        dt = (t_prev - t_i)
        v = predict_velocity_sd3(pipe, z_tar, t_i, emb_tar, cfg_tar)
        z_tar = z_tar + dt * v

    return z_tar

def try_call(obj, name):
    fn = getattr(obj, name, None)
    if callable(fn):
        fn()
        print(f"[OK] {name}")
    else:
        print(f"[SKIP] {name} not available")

def main():

    model_id = os.environ.get("SD3_MODEL_ID", "stabilityai/stable-diffusion-3-medium-diffusers")
    test_img = os.environ.get("TEST_IMG", "example_images/lighthouse.png")

    prompt_src = os.environ.get(
        "PROMPT_SRC",
        "The image features a tall white lighthouse standing prominently on a hill, with a beautiful blue sky in the background. The lighthouse is illuminated by a bright light, making it a prominent landmark in the scene."
    )
    prompt_tar = os.environ.get(
        "PROMPT_TAR",
        "The image features Big Ben clock tower standing prominently on a hill, with a beautiful blue sky in the background. The Big Ben clock tower is illuminated by a bright light, making it a prominent landmark in the scene."
    )

    T = int(os.environ.get("T", "50"))
    nmax = int(os.environ.get("NMAX", "33"))
    nmin = int(os.environ.get("NMIN", "0"))
    navg = int(os.environ.get("NAVG", "1"))
    cfg_src = float(os.environ.get("CFG_SRC", "3.5"))
    cfg_tar = float(os.environ.get("CFG_TAR", "13.5"))
    seed = int(os.environ.get("SEED", "0"))

    print(50*"-")
    print(f"Using seed: {seed} \n T: {T} \n nmax: {nmax} \n nmin: {nmin} \n navg: {navg} \n cfg_src: {cfg_src} \n cfg_tar: {cfg_tar}")
    print(50*"-")

    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    if device == "cuda":
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
    else:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            use_safetensors=True,
        )
    try_call(pipe, "enable_model_cpu_offload")
    try_call(pipe, "enable_attention_slicing")

    img = load_image(test_img, size=512)
    print(50*"-")
    print("Loaded image:", img)
    print(50*"-")
    x_src = encode_image_to_latents(pipe, img, device=device, dtype=dtype)
    print(50*"-")
    print("Coucou 1")
    print(50*"-")
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
    print(50*"-")
    print("Coucou 2")
    print(50*"-")
    out_img = decode_latents_to_image(pipe, z_out)
    out_img.save("flowedit_out.png")
    print("Saved: flowedit_out.png")

if __name__ == "__main__":
    main()
