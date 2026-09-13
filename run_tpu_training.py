import argparse
import subprocess
import sys
import os
import yaml
import tempfile
import time

def kill_hanging_tpu_processes():
    """Попытка освободить TPU, если он завис."""
    print("Проверка зависших TPU процессов...")
    try:
        # В Colab зависшие процессы python могут держать TPU
        # Это опасная команда, если у вас параллельно что-то важное обучается, 
        # но если мы одни, это спасет от Device busy
        # subprocess.run(["killall", "python3"], check=False) # Закомментировано, так как убьет текущий Colab cell
        pass
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description="NexusAI TPU Training Wrapper for Colab")
    parser.add_argument("--epochs", type=float, required=True, help="Количество эпох")
    parser.add_argument("--save_dir", type=str, default="mymodels", help="Папка для сохранения модели")
    parser.add_argument("--model_name", type=str, default="latest", help="Имя файла модели (без .pt)")
    parser.add_argument("--config", type=str, default="configs/100M.yaml", help="Путь к исходному конфигу")
    parser.add_argument("--data", type=str, nargs="+", default=["datasets/processed"], help="Пути к данным")
    
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. Читаем исходный конфиг
    if not os.path.exists(args.config):
        print(f"Ошибка: конфиг {args.config} не найден!")
        sys.exit(1)
        
    with open(args.config, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
        
    # 2. Модифицируем параметры
    if 'training' not in config_data:
        config_data['training'] = {}
        
    config_data['training']['epochs'] = args.epochs
    
    # Настраиваем пути сохранения (в зависимости от того, как ваш Trainer их читает)
    # Предполагается, что конфиг поддерживает checkpoint_dir или аналогичный параметр
    config_data['training']['checkpoint_dir'] = args.save_dir
    config_data['training']['checkpoint_name'] = f"{args.model_name}.pt"

    print(f"--- Подготовка к обучению на TPU ---")
    print(f"Эпохи: {args.epochs}")
    print(f"Сохранение в: {args.save_dir}/{args.model_name}.pt")
    
    # 3. Сохраняем временный конфиг
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_conf:
        yaml.dump(config_data, temp_conf)
        temp_config_path = temp_conf.name

    # 4. Формируем команду для запуска train.py
    cmd = [
        sys.executable, "train.py", 
        "--config", temp_config_path
    ]
    if args.data:
        cmd.extend(["--data"] + args.data)
        
    # Принудительно указываем XLA использовать TPU
    env = os.environ.copy()
    env["PJRT_DEVICE"] = "TPU"
    
    try:
        print(f"\nЗапуск: {' '.join(cmd)}\n")
        # Запускаем в подпроцессе, чтобы изолировать TPU XLA инициализацию
        process = subprocess.Popen(cmd, env=env)
        process.wait()
        
        if process.returncode == 0:
            print("\n--- ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО ---")
        else:
            print(f"\n--- ОШИБКА: процесс завершился с кодом {process.returncode} ---")
            print("Если вы видите 'Device or resource busy', перезапустите Runtime (Среду выполнения) в Colab: Runtime -> Restart session.")
            
    except KeyboardInterrupt:
        print("\nОбучение прервано пользователем. Убиваем процесс...")
        process.kill()
    finally:
        # Убираем временный конфиг
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

if __name__ == "__main__":
    main()
