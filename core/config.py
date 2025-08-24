from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class AppConfig:
    """Конфигурация приложения."""
    
    # Пути
    app_root: Path
    static_dir: Path
    logs_dir: Path
    temp_dir: Path
    exports_dir: Path
    
    # Настройки обработки
    supported_encodings: tuple[str, ...] = ('cp1251', 'utf-8', 'utf-8-sig')
    max_temp_files: int = 10
    
    # Настройки анализа
    default_tolerance_percent: float = 5.0
    min_work_threshold: float = 200.0
    
    # UI настройки
    default_window_size: tuple[int, int] = (1400, 900)
    log_level: str = "INFO"
    
    @classmethod
    def create_default(cls, app_root: Path = None) -> AppConfig:
        """Создает конфигурацию по умолчанию."""
        if app_root is None:
            app_root = Path(__file__).parent.parent
        
        return cls(
            app_root=app_root,
            static_dir=app_root / "static",
            logs_dir=app_root / "logs", 
            temp_dir=app_root / "temp",
            exports_dir=app_root / "exports",
        )
    
    def ensure_directories(self) -> None:
        """Создает необходимые директории."""
        for directory in [self.static_dir, self.logs_dir, self.temp_dir, self.exports_dir]:
            directory.mkdir(exist_ok=True)


# Глобальный экземпляр конфигурации
APP_CONFIG = AppConfig.create_default()