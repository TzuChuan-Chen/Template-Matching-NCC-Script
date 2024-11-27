import numpy as np
import cv2 as cv
import time
import os
import pandas as pd
from pathlib import Path


def compute_ncc(template, image_region):
    """
    計算正規化相關係數 (Normalized Cross-Correlation)
    
    參數:
        template: 模板影像
        image_region: 待比對的影像區域(大小需與模板相同)
    
    回傳:
        ncc: 正規化相關係數值 (-1 到 1 之間)
    """
    template = template.astype(np.float32)
    image_region = image_region.astype(np.float32)
    
    # 計算平均值
    template_mean = np.mean(template)
    image_region_mean = np.mean(image_region)
    
    # 計算零均值
    template_zero_mean = template - template_mean
    image_region_zero_mean = image_region - image_region_mean
    
    # 計算標準差
    template_std = np.std(template)
    image_region_std = np.std(image_region)
    
    # 計算NCC
    if template_std > 0 and image_region_std > 0:
        template_normalized = template_zero_mean / template_std
        image_region_normalized = image_region_zero_mean / image_region_std
        ncc = np.sum(template_normalized * image_region_normalized)
        ncc = ncc / (template_normalized.size)
    else:
        ncc = 0
        
    return ncc

def build_pyramid(image, levels):
    """
    建立影像金字塔
    
    參數:
        image: 輸入影像
        levels: 金字塔層數
    
    回傳:
        pyramid: 包含各層影像的列表
    """
    pyramid = [image]
    for i in range(levels - 1):
        pyramid.append(cv.pyrDown(pyramid[-1]))
    return pyramid

def match_template_traditional(image, template):
    """
    使用傳統NCC方法進行模板匹配
    
    參數:
        image: 搜索影像
        template: 模板影像
    
    回傳:
        best_loc: 最佳匹配位置 (x, y)
        score_map: NCC分數地圖
        execution_time: 執行時間
    """
    start_time = time.time()
    
    h, w = template.shape
    score_map = np.zeros((image.shape[0] - h + 1, image.shape[1] - w + 1))
    
    # 在每個位置計算NCC
    for y in range(score_map.shape[0]):
        for x in range(score_map.shape[1]):
            image_region = image[y:y+h, x:x+w]
            score_map[y, x] = compute_ncc(template, image_region)
    
    # 找到最佳匹配位置
    _, _, _, max_loc = cv.minMaxLoc(score_map)
    execution_time = time.time() - start_time
    
    return max_loc, score_map, execution_time
def find_multiple_patterns(image, templates, method='pyramid'):
    """
    在影像中尋找多個模板
    
    參數:
        image: 搜索影像
        templates: 字典，包含多個模板 {'name': template_image}
        method: 使用的匹配方法 ('traditional' 或 'pyramid')
    
    回傳:
        results: 字典，包含每個模板的匹配結果
        execution_times: 字典，包含每個模板的執行時間
    """
    results = {}
    execution_times = {}
    
    for name, template in templates.items():
        if method == 'pyramid':
            location, score_map, exec_time = match_template_pyramid(
                image, template)
        else:
            location, score_map, exec_time = match_template_traditional(
                image, template)
            
        results[name] = {
            'location': location,
            'score_map': score_map
        }
        execution_times[name] = exec_time
    
    return results, execution_times
def match_template_pyramid(image, template, pyramid_levels=3, search_window=20):
    """
    使用金字塔多尺度方法進行模板匹配
    
    參數:
        image: 搜索影像
        template: 模板影像
        pyramid_levels: 金字塔層數
        search_window: 局部搜索窗口大小
    
    回傳:
        best_loc: 最佳匹配位置 (x, y)
        score_map: 最後一層的NCC分數地圖
        execution_time: 執行時間
    """
    start_time = time.time()
    
    # 建立影像金字塔
    image_pyramid = build_pyramid(image, pyramid_levels)
    template_pyramid = build_pyramid(template, pyramid_levels)
    
    # 在最頂層進行完整搜索
    current_level = pyramid_levels - 1
    h, w = template_pyramid[current_level].shape
    score_map = np.zeros((
        image_pyramid[current_level].shape[0] - h + 1,
        image_pyramid[current_level].shape[1] - w + 1
    ))
    
    for y in range(score_map.shape[0]):
        for x in range(score_map.shape[1]):
            image_region = image_pyramid[current_level][y:y+h, x:x+w]
            score_map[y, x] = compute_ncc(template_pyramid[current_level], image_region)
    
    # 找到初始最佳位置
    _, _, _, max_loc = cv.minMaxLoc(score_map)
    best_x, best_y = max_loc
    
    # 逐層細化搜索
    for level in range(current_level - 1, -1, -1):
        # 計算在更精細尺度的搜索位置
        best_x *= 2
        best_y *= 2
        
        h, w = template_pyramid[level].shape
        score_map = np.zeros((2 * search_window + 1, 2 * search_window + 1))
        
        # 在局部視窗中搜索
        for dy in range(-search_window, search_window + 1):
            y = best_y + dy
            if y < 0 or y + h > image_pyramid[level].shape[0]:
                continue
                
            for dx in range(-search_window, search_window + 1):
                x = best_x + dx
                if x < 0 or x + w > image_pyramid[level].shape[1]:
                    continue
                
                image_region = image_pyramid[level][y:y+h, x:x+w]
                score_map[dy + search_window, dx + search_window] = compute_ncc(
                    template_pyramid[level], image_region)
        
        # 更新最佳匹配位置
        _, _, _, local_max_loc = cv.minMaxLoc(score_map)
        dy = local_max_loc[1] - search_window
        dx = local_max_loc[0] - search_window
        best_y += dy
        best_x += dx
    
    execution_time = time.time() - start_time
    return (best_x, best_y), score_map, execution_time

def calculate_center_point(location, template_shape):
    """
    計算模板匹配位置的中心點
    
    參數:
        location: 匹配位置的左上角座標 (x, y)
        template_shape: 模板的形狀 (height, width)
    
    回傳:
        center: 中心點座標 (x, y)
    """
    x, y = location
    h, w = template_shape
    center_x = x + w // 2
    center_y = y + h // 2
    return (center_x, center_y)

def calculate_displacement(center1, center2):
    """
    計算兩個中心點之間的位移
    
    參數:
        center1: 第一個中心點座標 (x, y)
        center2: 第二個中心點座標 (x, y)
    
    回傳:
        dx: X方向位移
        dy: Y方向位移
        distance: 歐氏距離
    """
    dx = center2[0] - center1[0]
    dy = center2[1] - center1[1]
    distance = np.sqrt(dx**2 + dy**2)
    return dx, dy, distance

def draw_all_results_with_centers(image, templates, results):
    """
    在影像上標記所有匹配結果和中心點
    
    參數:
        image: 原始影像
        templates: 字典，包含多個模板
        results: 匹配結果字典
    
    回傳:
        result_image: 標記後的影像
        centers: 字典，包含每個模板的中心點座標
    """
    result_image = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR)
    centers = {}
    
    # 為不同模板使用不同顏色
    colors = {
        'border': (0, 255, 0),  # 綠色表示對位框
        'center': (0, 0, 255)   # 紅色表示中央pattern
    }
    
    # 繪製每個模板的匹配結果和中心點
    for name, template in templates.items():
        location = results[name]['location']
        h, w = template.shape
        x, y = location
        
        # 繪製矩形框
        cv.rectangle(result_image, 
                    (x, y), 
                    (x + w, y + h), 
                    colors[name], 
                    2)
        
        # 計算並繪製中心點
        center = calculate_center_point(location, template.shape)
        centers[name] = center
        
        # 繪製十字標記
        cross_size = 10
        cx, cy = center
        cv.line(result_image, 
                (cx - cross_size, cy), 
                (cx + cross_size, cy), 
                colors[name], 
                2)
        cv.line(result_image, 
                (cx, cy - cross_size), 
                (cx, cy + cross_size), 
                colors[name], 
                2)
        
        # 在中心點旁加入標籤
        cv.putText(result_image, 
                  f"{name} ({cx}, {cy})", 
                  (cx + 15, cy), 
                  cv.FONT_HERSHEY_SIMPLEX, 
                  0.7, 
                  colors[name], 
                  2)
    
    # 如果有兩個中心點，則繪製連接線和距離標記
    if len(centers) == 2:
        border_center = centers['border']
        pattern_center = centers['center']
        
        # 繪製連接線
        cv.line(result_image, 
                border_center, 
                pattern_center, 
                (255, 255, 0), # 黃色
                1, 
                cv.LINE_AA)
        
        # 計算位移
        dx, dy, distance = calculate_displacement(border_center, pattern_center)
        
        # 在圖像上方顯示位移資訊
        info_y = 30
        cv.putText(result_image, 
                  f"dx: {dx:+.1f} pixels", 
                  (10, info_y), 
                  cv.FONT_HERSHEY_SIMPLEX, 
                  0.7, 
                  (255, 255, 0), 
                  2)
        cv.putText(result_image, 
                  f"dy: {dy:+.1f} pixels", 
                  (10, info_y + 25), 
                  cv.FONT_HERSHEY_SIMPLEX, 
                  0.7, 
                  (255, 255, 0), 
                  2)
        cv.putText(result_image, 
                  f"distance: {distance:.1f} pixels", 
                  (10, info_y + 50), 
                  cv.FONT_HERSHEY_SIMPLEX, 
                  0.7, 
                  (255, 255, 0), 
                  2)
    
    return result_image, centers

def process_single_image_with_comparison(image_path, templates, pattern_type):
    """
    使用兩種方法處理單張影像並比較結果
    
    參數:
        image_path: 影像檔案路徑
        templates: 模板字典
        pattern_type: 圖案類型 ('circle' 或 'cross')
    """
    # 讀取影像
    image = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
    if image is None:
        print(f"無法讀取影像: {image_path}")
        return None
    
    methods = ['traditional', 'pyramid']
    results_dict = {}
    
    for method in methods:
        # 執行模板匹配
        start_time = time.time()
        if method == 'traditional':
            results = {}
            execution_times = {}
            for name, template in templates.items():
                location, score_map, exec_time = match_template_traditional(image, template)
                results[name] = {
                    'location': location,
                    'score_map': score_map
                }
                execution_times[name] = exec_time
        else:
            results, execution_times = find_multiple_patterns(image, templates, method='pyramid')
        
        total_time = time.time() - start_time
        
        # 計算中心點
        centers = {}
        for name, result in results.items():
            location = result['location']
            template = templates[name]
            centers[name] = calculate_center_point(location, template.shape)
        
        # 計算位移
        dx, dy, distance = calculate_displacement(centers['border'], centers['center'])
        
        # 在結果圖上標記結果
        result_image = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR)
        
        # 為不同模板使用不同顏色
        colors = {
            'border': (0, 255, 0),  # 綠色
            'center': (0, 0, 255)   # 紅色
        }
        
        # 繪製每個模板的匹配結果和中心點
        for name, template in templates.items():
            location = results[name]['location']
            h, w = template.shape
            x, y = location
            
            # 繪製矩形框
            cv.rectangle(result_image, (x, y), (x + w, y + h), colors[name], 2)
            
            # 繪製中心點
            center = centers[name]
            cx, cy = center
            cross_size = 10
            cv.line(result_image, (cx - cross_size, cy), (cx + cross_size, cy), colors[name], 2)
            cv.line(result_image, (cx, cy - cross_size), (cx, cy + cross_size), colors[name], 2)
            
            # 顯示中心點座標
            cv.putText(result_image, f"{name} ({cx}, {cy})", 
                      (cx + 15, cy), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, colors[name], 2)
        
        # 繪製連接線
        border_center = centers['border']
        pattern_center = centers['center']
        cv.line(result_image, border_center, pattern_center, (255, 255, 0), 1)
        
        # 在圖上顯示位移資訊
        info_y = 30
        method_name = "Traditional" if method == "traditional" else "Pyramid"
        cv.putText(result_image, f"Method: {method_name}", 
                  (10, info_y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv.putText(result_image, f"dx: {dx:+.1f} pixels", 
                  (10, info_y + 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv.putText(result_image, f"dy: {dy:+.1f} pixels", 
                  (10, info_y + 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv.putText(result_image, f"distance: {distance:.1f} pixels", 
                  (10, info_y + 75), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv.putText(result_image, f"Time: {total_time:.3f} sec", 
                  (10, info_y + 100), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        results_dict[method] = {
            'image': result_image,
            'centers': centers,
            'displacement': (dx, dy, distance),
            'time': total_time
        }
    
    return results_dict

def batch_process_images(base_path):
    """批次處理所有影像"""
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    results_data = []
    pattern_types = ['circle', 'cross']
    
    for pattern_type in pattern_types:
        # 讀取對應的模板
        templates = {
            'border': cv.imread(str(Path(base_path) / 'pattern' / f'Template_Border{pattern_type.capitalize()}.bmp'), 
                              cv.IMREAD_GRAYSCALE),
            'center': cv.imread(str(Path(base_path) / 'pattern' / f'Template_{pattern_type}.bmp'), 
                              cv.IMREAD_GRAYSCALE)
        }
        
        # 處理該類型的所有影像
        pattern_dir = Path(base_path) / pattern_type
        image_files = sorted(pattern_dir.glob(f'Panel*_{pattern_type}*.bmp'))
        
        for image_path in image_files:
            print(f"處理: {image_path.name}")
            results = process_single_image_with_comparison(image_path, templates, pattern_type)
            
            if results is not None:
                # 儲存結果影像
                for method, result in results.items():
                    cv.imwrite(str(output_dir / f'{method}_{image_path.name}'), result['image'])
                
                # 記錄結果數據
                for method, result in results.items():
                    dx, dy, distance = result['displacement']
                    results_data.append({
                        'image_name': image_path.name,
                        'pattern_type': pattern_type,
                        'method': method,
                        'displacement_x': dx,
                        'displacement_y': dy,
                        'displacement_distance': distance,
                        'execution_time': result['time']
                    })
    
    # 轉換為DataFrame並進行分析
    df = pd.DataFrame(results_data)
    
    # 計算時間比較統計
    time_comparison = df.groupby(['method', 'pattern_type'])['execution_time'].agg(['mean', 'std', 'min', 'max']).round(3)
    
    # 儲存結果
    df.to_csv(output_dir / 'comparison_results.csv', index=False)
    time_comparison.to_csv(output_dir / 'time_comparison.csv')
    
    return df, time_comparison

def main():
    """主程式"""
    base_path = '.'  # 或使用實際的基礎路徑
    
    # 執行批次處理
    df, time_comparison = batch_process_images(base_path)
    
    # 顯示時間比較結果
    print("\n執行時間比較:")
    print(time_comparison)
    
    # 顯示加速比
    for pattern_type in df['pattern_type'].unique():
        traditional_time = df[(df['method'] == 'traditional') & 
                            (df['pattern_type'] == pattern_type)]['execution_time'].mean()
        pyramid_time = df[(df['method'] == 'pyramid') & 
                         (df['pattern_type'] == pattern_type)]['execution_time'].mean()
        speedup = traditional_time / pyramid_time
        print(f"\n{pattern_type} 圖案:")
        print(f"傳統方法平均時間: {traditional_time:.3f} 秒")
        print(f"金字塔方法平均時間: {pyramid_time:.3f} 秒")
        print(f"加速比: {speedup:.2f}x")

if __name__ == '__main__':
    main()