import os
import pandas as pd
import time

def try_to_csv(file_path, df=None, info="", index=False, header=False, mode='w', isPrintInfo=False):
    """
    安全写入 CSV 文件，如果文件/目录不存在会自动创建
    
    参数:
        file_path: 目标文件路径
        df: 要写入的 DataFrame (None 则创建空文件)
        info: 操作描述信息（用于打印）
        index: 是否写入索引
        header: 是否写入列名
        mode: 写入模式 ('w' 覆盖, 'a' 追加)
        isPrintInfo: 是否打印操作信息
    """
    # 检查文件夹是否存在，如果不存在则创建
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        if isPrintInfo:
            print(f"创建目录: {directory}")

    # 如果 df 为 None 且文件不存在，创建空 CSV
    if df is None and not os.path.exists(file_path):
        try:
            pd.DataFrame().to_csv(file_path, index=index, header=header)
            if isPrintInfo:
                print(f"创建空 CSV 文件: {file_path}")
            return
        except Exception as e:
            if isPrintInfo:
                print(f"创建空文件失败: {e}")
            raise

    # 正常写入数据
    if df is not None:
        while True:
            try:
                df.to_csv(file_path, index=index, header=header, mode=mode)
                if isPrintInfo:
                    print(f"成功写入 {info} 数据到: {file_path}")
                break
            except Exception as e:
                if isPrintInfo:
                    print(f"本次 {info} 数据写入 CSV 失败，尝试重新写入... 错误: {e}")
                time.sleep(0.1)  # 添加短暂延迟防止忙等待


def try_read_csv(file_path, info="", header=None, isPrintInfo=False):  
    while True:  
        try:  
            # 检查文件是否为空  
            if os.path.getsize(file_path) == 0:  
                if isPrintInfo: print(f"{info}文件为空。")
                return pd.DataFrame()
            # 读取文件的前几行以检查是否有有效数据  
            with open(file_path, 'r') as f:  
                first_line = f.readline().strip()  
                if not first_line:  
                    if isPrintInfo: print(f"{info}文件没有有效数据。")
                    return pd.DataFrame()      
            csv = pd.read_csv(file_path, header=header)
            # print(csv)  
            return csv  
        except Exception as e:  
            if isPrintInfo:  
                print(f"本次{info}读取csv失败, 错误信息: {e}，尝试重新读取...")