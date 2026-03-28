def first_stage_batch(images, processor, model, max_tokens=300):
    conversations = []
    for image in images:
        conversations.append([
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe the image in great detail."},
                ],
            },
        ])

    inputs = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
        padding=True,
    ).to(model.device)
    
    with __import__('torch').no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
        
        # We need to extract just the new generated tokens, or decode the whole and let the model figure it out?
        # The apply_chat_template output includes the prompt tokens.
        # But wait, batch_decode on outputs will just decode the whole string including the prompt.
        # In first_stage(), it just takes `processor.batch_decode(outputs, skip_special_tokens=True)[0]`
        assistant_texts = processor.batch_decode(outputs, skip_special_tokens=True)
        # Wait, for LLaVa/LFM-VL we might just get the assistant output or the whole text. 
        # If skip_special_tokens=True, it strips <|endoftext|>.
        # Actually in first_stage, `outputs` is decoded directly. Let's see what first_stage does exactly.
