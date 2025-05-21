# Запуск среды и обучения агентов
Нужно установить и запустить Docker-контейнер с симулятором.
Перед этим создать папки `starpu_home`, `cache_hf` и `proj_dir` в директории, в которой запускается контейнер.
```bash
sudo docker pull sivtsovdt/graph_agent:v2-iter-based
# Для систем с GPU (необходим пакет nvidia-container-toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installation-guide)
# sudo docker run --gpus all --network host -v starpu_home:/home/jovyan/sivtsov -v cache_hf:/workspace/cache_hf -v proj_dir:/workspace/proj_dir -it sivtsovdt/graph_agent:v2-iter-based -- /bin/bash
# Для систем без GPU
sudo docker run --network host -v starpu_home:/home/jovyan/sivtsov -v cache_hf:/workspace/cache_hf -v proj_dir:/workspace/proj_dir -it sivtsovdt/graph_agent:v2-iter-based -- /bin/bash
```
В контейнере запустить Jupyter Lab
```bash
jupyter lab --allow-root
```
И скопировать в директорию `workspace` файл `train_agent.ipynb`
## Постановка задачи
Более подробное описание [тут](https://github.com/c0lbarator/rl-sched-project/blob/cbf0b7ad5265da454c88da9350d02c84c4f5d5a1/%D0%9C%D0%B0%D1%82__%D0%BF%D0%BE%D1%81%D1%82%D0%B0%D0%BD%D0%BE%D0%B2%D0%BA%D0%B0_%D0%B7%D0%B0%D0%B4%D0%B0%D1%87%D0%B8%20(4).pdf)
### Среда (Environment)
Наша среда — это GraphRunner и сам вычислительный граф.

### Состояние (State)
Информация, на основе которой агент принимает решение. Это может быть:
- Характеристики текущего узла графа (тип операции, ожидаемое время выполнения на CPU/GPU)
- Состояние зависимостей узла (готовы ли входные данные)
- Общее состояние графа (сколько узлов осталось, сколько уже запланировано)

**Упрощение:**  
Начнем с простого: Состояние будет представлять характеристики одного конкретного узла, который мы рассматриваем для планирования.

### Действие (Action)
Решение, которое принимает агент для конкретного узла. В нашем случае это:
- **Приоритет (Priority):** Числовое значение. Чем выше, тем раньше StarPU постарается выполнить задачу.
- **Сродство к устройству (Device Affinity):** На каких типах устройств (CPU/GPU) может выполняться задача. Агент может выбрать:
  - CPU
  - GPU
  - Оба типа (mixed)


### Награда (Reward)
Сигнал обратной связи, показывающий, насколько хорошо агент справился. Логично использовать отрицательное время выполнения всего графа (-timings). Чем меньше время, тем больше награда.

### Компоненты обучения
- **Политика (Policy - Actor):** Нейронная сеть, которая по состоянию предсказывает распределение вероятностей для действий (например, вероятность выбора CPU или GPU).
- **Оценщик ценности (Value Function - Critic):** Нейронная сеть, которая по состоянию предсказывает ожидаемую будущую награду (насколько "хорошо" текущее состояние).

### Алгоритм PPO
Использует Actor и Critic для обновления политики таким образом, чтобы максимизировать награду, избегая слишком больших изменений политики на каждом шаге (отсюда "Proximal").
