Запустить контейнеры в фоне:
docker compose up -d

Посмотреть состояние:
docker compose ps

Остановить и удалить контейнеры, сохранив данные MinIO:
docker compose down

Остановить без удаления контейнеров:
docker compose stop

Снова запустить остановленные контейнеры:
docker compose start