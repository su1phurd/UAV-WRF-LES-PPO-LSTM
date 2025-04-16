from netCDF4 import Dataset
import numpy as np

with Dataset('training_data.nc', 'r') as nc, open('nc_info.txt', 'w', encoding='utf-8') as f:
    f.write("维度:\n")
    for dim in nc.dimensions:
        f.write(f"  {dim}: {len(nc.dimensions[dim])}\n")
    f.write("\n变量:\n")
    for var in nc.variables:
        v = nc.variables[var]
        f.write(f"  {var}: shape={v.shape}, dtype={v.dtype}\n")
        for attr in v.ncattrs():
            f.write(f"    {attr}: {getattr(v, attr)}\n")
        # 只对数值型变量统计范围
        try:
            arr = v[:].compressed() if hasattr(v[:], 'compressed') else v[:]
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                f.write(f"    min={np.nanmin(arr)}, max={np.nanmax(arr)}\n")
        except Exception as e:
            f.write(f"    (无法统计数值范围: {e})\n")
    f.write("\n文件检查完毕。\n")
print("nc_info.txt已生成，包含所有维度、变量、属性和数值范围信息。")
