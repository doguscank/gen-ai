from gen_ai.tasks.image_gen.lora.lora_manager import LoraManager

if __name__ == "__main__":
    lora_manager = LoraManager(
        lora_dir="E:\\Scripting Workspace\\Python\\GenAI\\gen-ai\\lora\\sd_15",
        auto_register=True,
    )

    print(lora_manager.models)

    lora = lora_manager.get_model_by_name("70s_Horror_Movie")
    lora.set_loaded()

    print(lora_manager.models)
