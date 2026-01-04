import torch

from minimal_internvl_inference import load_model
from internvl_skeleton_adapter import InternVLWithSkeleton


def main():
    # Load frozen base InternVL (1B) model
    tokenizer, base_model, device = load_model()

    # Wrap with skeleton adapter
    skel_model = InternVLWithSkeleton(base_model).to(device)
    skel_model.eval()

    # Dummy skeleton features: batch=1, T=10 frames, 46-dim per frame
    skeleton_feats = torch.randn(1, 10, 46, device=device)

    prompt = (
        "Given the patient's gait skeleton parameters, briefly describe any abnormal "
        "gait patterns you observe."
    )

    with torch.no_grad():
        output_ids = skel_model.generate_with_skeleton(
            tokenizer=tokenizer,
            prompt=prompt,
            skeleton_feats=skeleton_feats,
            max_new_tokens=64,
            do_sample=False,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\n=== Skeleton + text demo ===")
    print("Prompt:", prompt)
    print("\nModel output:\n", text)


if __name__ == "__main__":
    main()






