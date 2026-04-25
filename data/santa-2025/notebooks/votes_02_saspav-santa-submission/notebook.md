# Santa-submission

- **Author:** Sas Pav
- **Votes:** 602
- **Ref:** saspav/santa-submission
- **URL:** https://www.kaggle.com/code/saspav/santa-submission
- **Last run:** 2026-04-24 19:16:03.643000

---

## Acknowledgments

This notebook is a Frankenstein of
[**Why Not**](https://www.kaggle.com/code/jazivxt/why-not/notebook) by **jazivxt** and [**Santa 2025 - fix direction**](https://www.kaggle.com/code/chistyakov/santa-2025-fix-direction) & [**Shake, Shake, Shake**](https://www.kaggle.com/code/chistyakov/shake-shake-shake) by **Stanislav Chistyakov**

---
To use full optimization, you must set the parameter:

DEBUG = False

```python
DEBUG = False

MAX_HOURS = 11.7
```

```python
from pathlib import Path
from shutil import copy

donor_file = "/kaggle/input/santa-2025-csv/santa-2025.csv"
if Path('/kaggle/input/datasets/saspav/santa-2025-best/submission.csv').is_file():
    print('Файл из приватного датасета')
    donor_file = '/kaggle/input/datasets/saspav/santa-2025-best/submission.csv'
copy(donor_file, '/kaggle/working/submission.csv')
copy('/kaggle/input/santa-2025-csv/shake_public', '/kaggle/working/')
copy('/kaggle/input/santa-2025-csv/bbox3', '/kaggle/working/')
```

```python
!chmod +x ./bbox3
!chmod +x ./shake_public
```

```python
import numpy as np
import pandas as pd
from decimal import Decimal, getcontext
from shapely import affinity, touches
from shapely.geometry import Polygon
from shapely.ops import unary_union
from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar

getcontext().prec = 30
scale_factor = 1


class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

    def __init__(self, center_x='0', center_y='0', angle='0'):
        """Initializes the Christmas tree with a specific position and rotation."""
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)

        trunk_w = Decimal('0.15')
        trunk_h = Decimal('0.2')
        base_w = Decimal('0.7')
        mid_w = Decimal('0.4')
        top_w = Decimal('0.25')
        tip_y = Decimal('0.8')
        tier_1_y = Decimal('0.5')
        tier_2_y = Decimal('0.25')
        base_y = Decimal('0.0')
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon(
            [
                # Start at Tip
                (Decimal('0.0') * scale_factor, tip_y * scale_factor),
                # Right side - Top Tier
                (top_w / Decimal('2') * scale_factor, tier_1_y * scale_factor),
                (top_w / Decimal('4') * scale_factor, tier_1_y * scale_factor),
                # Right side - Middle Tier
                (mid_w / Decimal('2') * scale_factor, tier_2_y * scale_factor),
                (mid_w / Decimal('4') * scale_factor, tier_2_y * scale_factor),
                # Right side - Bottom Tier
                (base_w / Decimal('2') * scale_factor, base_y * scale_factor),
                # Right Trunk
                (trunk_w / Decimal('2') * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal('2') * scale_factor, trunk_bottom_y * scale_factor),
                # Left Trunk
                (-(trunk_w / Decimal('2')) * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                # Left side - Bottom Tier
                (-(base_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                # Left side - Middle Tier
                (-(mid_w / Decimal('4')) * scale_factor, tier_2_y * scale_factor),
                (-(mid_w / Decimal('2')) * scale_factor, tier_2_y * scale_factor),
                # Left side - Top Tier
                (-(top_w / Decimal('4')) * scale_factor, tier_1_y * scale_factor),
                (-(top_w / Decimal('2')) * scale_factor, tier_1_y * scale_factor),
            ]
        )
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(rotated,
                                          xoff=float(self.center_x * scale_factor),
                                          yoff=float(self.center_y * scale_factor))

    def clone(self) -> "ChristmasTree":
        return ChristmasTree(
            center_x=str(self.center_x),
            center_y=str(self.center_y),
            angle=str(self.angle),
        )


def get_tree_list_side_lenght(tree_list: list[ChristmasTree]) -> Decimal:
    all_polygons = [t.polygon for t in tree_list]
    bounds = unary_union(all_polygons).bounds
    return Decimal(max(bounds[2] - bounds[0], bounds[3] - bounds[1])) / scale_factor


def get_total_score(dict_of_side_length: dict[str, Decimal]):
    score = 0
    for k, v in dict_of_side_length.items():
        score += v ** 2 / Decimal(k)
    return score


def parse_csv(csv_path) -> dict[str, list[ChristmasTree]]:
    print(f'\nparse_csv: {csv_path=}')

    result = pd.read_csv(csv_path)
    result['x'] = result['x'].str.strip('s')
    result['y'] = result['y'].str.strip('s')
    result['deg'] = result['deg'].str.strip('s')
    result[['group_id', 'item_id']] = result['id'].str.split('_', n=2, expand=True)

    dict_of_tree_list = {}
    dict_of_side_length = {}
    for group_id, group_data in result.groupby('group_id'):
        tree_list = [ChristmasTree(center_x=row['x'], center_y=row['y'], angle=row['deg'])
                     for _, row in group_data.iterrows()]
        dict_of_tree_list[group_id] = tree_list
        dict_of_side_length[group_id] = get_tree_list_side_lenght(tree_list)

    return dict_of_tree_list, dict_of_side_length


def calculate_bbox_side_at_angle(angle_deg, points):
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix_T = np.array([[c, s], [-s, c]])
    rotated_points = points.dot(rot_matrix_T)
    min_xy = np.min(rotated_points, axis=0);
    max_xy = np.max(rotated_points, axis=0)
    return max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1])


def optimize_rotation(trees):
    all_points = []
    for tree in trees: all_points.extend(list(tree.polygon.exterior.coords))
    points_np = np.array(all_points)

    hull_points = points_np[ConvexHull(points_np).vertices]

    initial_side = calculate_bbox_side_at_angle(0, hull_points)

    res = minimize_scalar(lambda a: calculate_bbox_side_at_angle(a, hull_points),
                          bounds=(0.001, 89.999), method='bounded')
    found_angle_deg = res.x
    found_side = res.fun

    improvement = initial_side - found_side

    EPSILON = 1e-10

    if improvement > EPSILON:
        best_angle_deg = found_angle_deg
        best_side = Decimal(found_side) / scale_factor
    else:
        best_angle_deg = 0.0
        best_side = Decimal(initial_side) / scale_factor

    return best_side, best_angle_deg


def apply_rotation(trees, angle_deg):
    if not trees or abs(angle_deg) < 1e-9: return [t.clone() for t in trees]

    bounds = [t.polygon.bounds for t in trees]
    min_x = min(b[0] for b in bounds);
    min_y = min(b[1] for b in bounds)
    max_x = max(b[2] for b in bounds);
    max_y = max(b[3] for b in bounds)
    rotation_center = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0])

    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix = np.array([[c, -s], [s, c]])

    points = np.array([[float(t.center_x), float(t.center_y)] for t in trees])
    shifted = points - rotation_center
    rotated = shifted.dot(rot_matrix.T) + rotation_center

    rotated_trees = []
    for i in range(len(trees)):
        new_tree = ChristmasTree(Decimal(rotated[i, 0]), Decimal(rotated[i, 1]),
                                 Decimal(trees[i].angle + Decimal(angle_deg)))
        rotated_trees.append(new_tree)
    return rotated_trees


def fix_direction(current_solution_path='submission.csv', out_file='submission.csv'):
    # Load current best solution
    dict_of_tree_list, dict_of_side_length = parse_csv(current_solution_path)

    # Calculate current total score
    current_score = get_total_score(dict_of_side_length)
    print(f'{current_score=:0.12f}')

    initial_trees = [
        ChristmasTree(1, 0, 0),  # Смотрит вправо (0°)
        ChristmasTree(0, 1, 90),  # Смотрит вверх (90°)
        ChristmasTree(-1, 0, 180),  # Смотрит влево (180°)
        ChristmasTree(0, -1, 270)  # Смотрит вниз (270°)
    ]

    best_side, best_angle_deg = optimize_rotation(initial_trees)
    fixed_trees = apply_rotation(initial_trees, best_angle_deg)

    for group_id_main in range(200, 2, -1):
        group_id_main = f'{int(group_id_main):03n}'

        initial_trees = dict_of_tree_list[group_id_main]
        best_side, best_angle_deg = optimize_rotation(initial_trees)
        fixed_trees = apply_rotation(initial_trees, best_angle_deg)

        cur_side = dict_of_side_length[group_id_main]
        if best_side < cur_side:
            print(f'n={int(group_id_main)}, {best_side:0.8f}-> {cur_side:0.8f} '
                  f'({best_side - cur_side:0.8f})')

            dict_of_tree_list[group_id_main] = fixed_trees
            dict_of_side_length[group_id_main] = best_side

    new_score = get_total_score(dict_of_side_length)
    diff_score = current_score - new_score
    print(f'    {new_score=:0.12f}\n'
          f'    {diff_score=:0.12f}\n')

    if diff_score > 0:
        print('Достигнут прогресс --> сохраняю результат')
        tree_data = []
        for group_name, tree_list in dict_of_tree_list.items():
            for item_id, tree in enumerate(tree_list):
                tree_data.append({
                    'id': f'{group_name}_{item_id}',
                    'x': f's{tree.center_x}',
                    'y': f's{tree.center_y}',
                    'deg': f's{tree.angle}'
                })
        tree_data = pd.DataFrame(tree_data)
        tree_data.to_csv(out_file, index=False)
    
    return current_score, new_score
```

```python
import os
import time
import subprocess
import threading
from shutil import copy
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed



def run_bbox_simple_with_timeout(debug=False):
    """
    Запуск экспериментов с ограничением по времени.
    Если с момента начала прошло более MAX_HOURS часов - выход.
    """
    # Создаем директории
    os.makedirs("bbox_sub", exist_ok=True)
    
    # Лог-файл
    log_file = "bbox_experiments.log"
    
    # Запоминаем время начала
    start_time = datetime.now()
    timeout = timedelta(hours=MAX_HOURS)
    
    print(f"Начало экспериментов в: {start_time}")
    print(f"Таймаут через: {MAX_HOURS} часов (до {start_time + timeout})")
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Начало: {start_time}\n")
        f.write(f"Таймаут: {MAX_HOURS} часов\n")
        f.write('='*50 + '\n')
        
        total_runs = 0
        completed_runs = 0

        # Считаем общее количество запусков
        n_min = 100
        n_max = 2000
        r_min = 10        
        r_max = 200
        n_values = list(range(n_min, n_max + 1, 100))
        r_values = list(range(r_min, r_max + 1, 10))
        total_runs = len(n_values) * len(r_values)
        
        print(f"Всего планируется запусков: {total_runs}")

        if not debug:
            initial_score, final_score = fix_direction()
                
        for r_value in r_values:

            if debug:
                break

            for n_value in n_values:
                
                # Проверяем не истекло ли время
                current_time = datetime.now()
                elapsed = current_time - start_time
                
                if elapsed > timeout:
                    print(f"\n⏰ ВРЕМЯ ИСТЕКЛО! Прошло {elapsed}")
                    print(f"Завершаем выполнение...")
                    f.write(f"\n⏰ ВРЕМЯ ИСТЕКЛО! Прошло {elapsed}\n")
                    f.write(f"Завершаем выполнение...\n")
                    return
                
                # Выводим прогресс
                completed_runs += 1
                progress = (completed_runs / total_runs) * 100
                time_left = (timeout - elapsed) / (completed_runs) * (total_runs - completed_runs) if completed_runs > 0 else timeout
                
                print(f"[Прогресс: {progress:.1f}%] [Прошло: {elapsed}] [Осталось: ~{time_left}]")
                print(f"Итерация {completed_runs} - Параметры: n={n_value}, r={r_value}")
                
                f.write(f"\n[Время: {current_time}] [Прошло: {elapsed}]\n")
                f.write(f"Итерация {completed_runs} - Параметры: n={n_value}, r={r_value}\n")
                
                try:
                    # Запускаем команду
                    result = subprocess.run(
                        ["./bbox3", "-n", str(n_value), "-r", str(r_value)],
                        capture_output=True,
                        text=True,
                        timeout=1200  # Таймаут 20 минут на одну итерацию
                    )
                    
                    # Выводим результат
                    print(result.stdout)
                    f.write(result.stdout + "\n")
                    
                    if result.stderr:
                        print("Ошибки:", result.stderr)
                        f.write(f"Ошибки: {result.stderr}\n")
                    
                except subprocess.TimeoutExpired:
                    error_msg = f"⚠ Таймаут команды (20 минут) для n={n_value}, r={r_value}, i={completed_runs}"
                    print(error_msg)
                    f.write(error_msg + "\n")
                    continue
                    
                except Exception as e:
                    error_msg = f"❌ Ошибка при запуске: {e}"
                    print(error_msg)
                    f.write(error_msg + "\n")
                    continue
                
                # Сохраняем файл
                if os.path.exists("submission.csv"):
                    new_name = f"bbox_sub/submi-n{n_value}_r{r_value}_i{completed_runs}.csv"
                    try:
                        copy("submission.csv", new_name)
                        success_msg = f"✓ Сохранено: {new_name}"
                        print(success_msg)
                        f.write(success_msg + "\n")
                    except Exception as e:
                        error_msg = f"❌ Ошибка при сохранении файла: {e}"
                        print(error_msg)
                        f.write(error_msg + "\n")
                
                print("---")
                f.write("---\n")
                f.flush()

                if not debug:
                    _, final_score = fix_direction()

                if debug:
                    break
                    
            if debug:
                break
        
        # Если дошли до конца
        end_time = datetime.now()
        total_elapsed = end_time - start_time
        print(f"\n{'='*50}")
        print(f"✅ ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
        print(f"Начало: {start_time}")
        print(f"Завершение: {end_time}")
        print(f"Общее время: {total_elapsed}")
        print(f"Выполнено запусков: {completed_runs} из {total_runs}")
        if not debug:
            print(f"Начальная метрика: {initial_score:.12f}")
            print(f"Финальная метрика: {final_score:.12f}")
            print(f"Прирост метрики:    {initial_score - final_score:.12f}")
        print('='*50)
        
        f.write(f"\n{'='*50}\n")
        f.write(f"✅ ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!\n")
        f.write(f"Начало: {start_time}\n")
        f.write(f"Завершение: {end_time}\n")
        f.write(f"Общее время: {total_elapsed}\n")
        f.write(f"Выполнено запусков: {completed_runs} из {total_runs}\n")
        if not debug:
            f.write(f"Начальная метрика: {initial_score:.12f}")
            f.write(f"Финальная метрика: {final_score:.12f}")
            f.write(f"Прирост метрики:   {initial_score - final_score:.12f}")        
        f.write('='*50 + '\n')
    
    print("Эксперименты завершены!")


run_bbox_simple_with_timeout(debug=DEBUG)
```

```python
import pandas as pd
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree

# Set precision for Decimal (25 is good for contest standards)
getcontext().prec = 25
scale_factor = Decimal("1e18")


class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""
    
    def __init__(self, center_x="0", center_y="0", angle="0"):
        """Initializes the Christmas tree with a specific position and rotation."""
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)
        
        # Tree dimensions
        trunk_w = Decimal("0.15")
        trunk_h = Decimal("0.2")
        base_w = Decimal("0.7")
        mid_w = Decimal("0.4")
        top_w = Decimal("0.25")
        tip_y = Decimal("0.8")
        tier_1_y = Decimal("0.5")
        tier_2_y = Decimal("0.25")
        base_y = Decimal("0.0")
        trunk_bottom_y = -trunk_h
        
        # Define the 15 vertices of the tree polygon
        initial_polygon = Polygon([
            (Decimal("0.0") * scale_factor, tip_y * scale_factor),
            (top_w / Decimal("2") * scale_factor, tier_1_y * scale_factor),
            (top_w / Decimal("4") * scale_factor, tier_1_y * scale_factor),
            (mid_w / Decimal("2") * scale_factor, tier_2_y * scale_factor),
            (mid_w / Decimal("4") * scale_factor, tier_2_y * scale_factor),
            (base_w / Decimal("2") * scale_factor, base_y * scale_factor),
            (trunk_w / Decimal("2") * scale_factor, base_y * scale_factor),
            (trunk_w / Decimal("2") * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / Decimal("2")) * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / Decimal("2")) * scale_factor, base_y * scale_factor),
            (-(base_w / Decimal("2")) * scale_factor, base_y * scale_factor),
            (-(mid_w / Decimal("4")) * scale_factor, tier_2_y * scale_factor),
            (-(mid_w / Decimal("2")) * scale_factor, tier_2_y * scale_factor),
            (-(top_w / Decimal("4")) * scale_factor, tier_1_y * scale_factor),
            (-(top_w / Decimal("2")) * scale_factor, tier_1_y * scale_factor),
        ])
        
        # Apply rotation and translation
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(
            rotated,
            xoff=float(self.center_x * scale_factor),
            yoff=float(self.center_y * scale_factor)
        )


def has_overlap(trees):
    """Check if any two ChristmasTree polygons overlap."""
    if len(trees) <= 1:
        return False
    
    polygons = [t.polygon for t in trees]
    tree_index = STRtree(polygons)
    
    for i, poly in enumerate(polygons):
        indices = tree_index.query(poly)
        for idx in indices:
            if idx == i:
                continue
            if poly.intersects(polygons[idx]) and not poly.touches(polygons[idx]):
                return True
    return False


def load_trees_for_n(n, df):
    """Load all trees for a given N from the submission DataFrame."""
    group_data = df[df["id"].str.startswith(f"{n:03d}_")]
    trees = []
    for _, row in group_data.iterrows():
        x = str(row["x"]).lstrip('s')
        y = str(row["y"]).lstrip('s')
        deg = str(row["deg"]).lstrip('s')
        if x and y and deg:
            trees.append(ChristmasTree(x, y, deg))
    return trees


def find_invalid_groups(new_csv_path, max_n=200):
    df_new = pd.read_csv(new_csv_path)
    
    replaced_n = []
    
    for n in range(1, max_n + 1):
        trees = load_trees_for_n(n, df_new)
        if trees and has_overlap(trees):
            replaced_n.append(n)
    
    return replaced_n
```

```python
# Сохраняем файл
if os.path.exists("submission.csv"):
    new_name = "bbox_sub/submission_new.csv"
    try:
        copy("submission.csv", new_name)
        success_msg = f"✓ Сохранено: {new_name}"
        print(success_msg)
    except Exception as e:
        error_msg = f"❌ Ошибка при сохранении файла: {e}"
        print(error_msg)
```

```python
!./shake_public --input="submission.csv" --output="submission.csv"
```

```python
import csv


def load_groups(filename):
    """
    Загружает файл в словарь:
    {
        '001': [строка1, строка2],
        '002': [...],
        ...
    }
    """
    groups = {}
    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # сохраняем заголовок
        for row in reader:
            full_id = row[0]
            group = full_id.split('_')[0]

            groups.setdefault(group, []).append(row)

    return header, groups


def replace_group(target_file, donor_file, group_id, output_file=None):
    """
    target_file – файл, в котором меняем группу
    donor_file  – эталонный файл-источник
    group_id    – '004'
    output_file – куда сохранить (если None – перезапись target_file)
    """
    if output_file is None:
        output_file = target_file

    # Загружаем оба файла
    header_t, groups_t = load_groups(target_file)
    header_d, groups_d = load_groups(donor_file)

    # if header_t != header_d:
    #     raise ValueError("Ошибка: заголовки файлов отличаются!")

    if group_id not in groups_d:
        raise ValueError(f"В файле-донора нет группы {group_id}")

    # Заменяем
    groups_t[group_id] = groups_d[group_id]

    # Сохраняем результат
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header_t)

        # сортируем группы по номеру, чтобы порядок не сломать
        for g in sorted(groups_t.keys(), key=lambda x: int(x)):
            for row in groups_t[g]:
                writer.writerow(row)

    print(f"✔ Группа {group_id} заменена и сохранена в {output_file}")
```

```python
# Сохраняем файл
if os.path.exists("submission.csv"):
    new_name = "bbox_sub/submission_shake.csv"
    try:
        copy("submission.csv", new_name)
        success_msg = f"✓ Сохранено: {new_name}"
        print(success_msg)
    except Exception as e:
        error_msg = f"❌ Ошибка при сохранении файла: {e}"
        print(error_msg)

GROIP_IDXS = find_invalid_groups("submission.csv", max_n=200)
if GROIP_IDXS:
    for GROIP_ID in GROIP_IDXS:
        replace_group(
            target_file="submission.csv",
            donor_file=donor_file,
            group_id=f'{GROIP_ID:03d}',
            output_file="submission.csv"
        )
```

```python
from IPython.display import display, FileLink
from zipfile import ZipFile, ZIP_DEFLATED as ZD
from datetime import datetime
from glob import glob

files = glob(f'*.csv') + glob(f'*.log') + glob(f'bbox_sub/*.csv')
formatted_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
zip_filename = f'kaggle_bbox_{formatted_time}.zip'
with ZipFile(zip_filename, 'w',  compression=ZD, compresslevel=9) as zip_file:
    for filename in files:
        print(filename)
        zip_file.write(filename)
FileLink(zip_filename)
```