import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. 读取数据
file_path = '/home/yuanhongxu/tmp/cpc/MIMICIIIdata_Circulatory_MultiClass_Selected.csv'
df = pd.read_csv(file_path)

# 2. 确定特征列和患者ID列
label_columns = [
    'Acute cerebrovascular disease',
    'Acute myocardial infarction',
    'Cardiac dysrhythmias',
    'Congestive heart failure; nonhypertensive',
    'Coronary atherosclerosis and other heart disease',
    'Hypertension with complications and secondary hypertension'
]
id_column = 'PatientID'
feature_columns = [col for col in df.columns if col not in label_columns + [id_column]]

# 3. 创建一个存储患者第一主成分序列的列表
results = []

# 4. 按患者ID分组，分别计算每个患者的Kernel PCA主成分序列
for patient_id, group in df.groupby(id_column):
    # 提取当前患者的特征数据
    features = group[feature_columns]

    # 如果该患者的数据不足以计算主成分（例如只有一行数据），跳过
    if features.shape[0] < 2:
        print(f"患者 {patient_id} 数据不足，无法计算主成分，跳过...")
        continue

    # 标准化数据
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # 计算Kernel PCA第一主成分
    kpca = PCA(n_components=1)
    first_pc = kpca.fit_transform(scaled_features).flatten()  # 展平为一维数组

    # 保存每个患者的主成分序列
    results.append({
        'PatientID': patient_id,
        'FirstPrincipalComponent': list(first_pc)  # 将主成分序列存为列表
    })

# 5. 转换为DataFrame格式，每个患者一行
results_df = pd.DataFrame(results)

# 6. 保存结果到新的CSV文件
output_file = '/home/yuanhongxu/tmp/cpc/patientwise_first_principal_component_sequence.csv'
results_df.to_csv(output_file, index=False)

# 打印结果预览
print("结果已保存为:", output_file)
print(results_df.head())
