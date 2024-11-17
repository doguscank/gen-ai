from gen_ai.image_gen.lora.lora_manager import LoraManager

if __name__ == "__main__":
    lora_manager = LoraManager(
        lora_dir="E:\\Scripting Workspace\\Python\\GenAI\\gen-ai\\lora\\sd_15",
        auto_register=True,
    )
    print(lora_manager.registered_models)
