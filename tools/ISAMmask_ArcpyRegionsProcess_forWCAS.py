import os
import arcpy
import rasterio
import f90nml
from arcpy.ia import RasterCalculator
from arcpy.sa import *
from osgeo import gdal


def ISAMmask_ArcpyRegionsProcess(
    Workdir = "x/x/", #
    mask_shapes_dir = "x/x/", #
    GRIDCRO2D_temptif = "", #
    shapenames = [],#
    shapenames_var = [], #
):
    """
    :param Workdir: arcpy处理过程存放中间文件的文件夹
    :param mask_shapes_dir: 制作mask的shp文件所在的文件夹，必须为只有一个闭合区域的shp，一个shp一个mask
    :param GRIDCRO2D_temptif: GRIDCRO2D的框架文件路径，可通过var2tiff生成temp的过程生成
    :param shapenames: shp文件的名称，
    :param shapenames_var: shp生成的maskvar的变量名，不能超过3个字母！！，与上面shp文件名一一对应
    :return:
    """
    # mask_shapes_dir = r"E:\ArcGISFiles\yaan_ISAMMASK\county_shp\\"  # 区域shp所在目录
    PolygonToRaster_shapes_dir = Workdir+r"PolygonToRaster\\"
    RasterCalculator_dir = Workdir+r"RasterCalculator\\"
    RasterReclass_dir = Workdir+r"RasterReclass\\"
    Mosaic_dir = Workdir+r"Mosaic\\"
    Regins_tiff_dir = Workdir+r"Regionstifs\\"  # 溯源区域tiff所在dir，为此过程的最终生成的结果

    # GRIDCRO2D_temptif = r"E:\CMAQdata_yaan202306\temp.tif"  # CMAQ区域网格，ISAMMASK的框架
    # GRIDDECfile_dir = r"E:\CMAQdata_yaan202306\GRIDDESC_d02"
    # ISAM_region_file_dir = r"E:\CMAQdata_yaan202306\ISAM_REGION.nc"

    # shapenames = ['巴中','成都','达州','德阳','广安','广元','乐山','泸州','眉山','绵阳','南充','内江','遂宁','宜宾','资阳','自贡','雅安']
    # shapenames_var = ['BZ','CD','DZ','DY','GA','GY','LS','LZ','MS','MY','NC','NJ','SN','YB','ZY','ZG','YA']
    # shapenames = ['CD', 'DY', 'LS', 'MS', 'MY', 'NJ', 'SN', 'YB', 'ZY', 'ZG', 'YA']  # shp文件名称
    # shapenames_var = ['CD', 'DY', 'LS', 'MS', 'MY', 'NJ', 'SN', 'YB', 'ZY', 'ZG', 'YA']  # 对应VAR

    mask_shapes_pre = os.listdir(mask_shapes_dir)
    mask_shapes = []
    for f in mask_shapes_pre:
        if f.split('.')[1] == 'shp':
            mask_shapes.append(f)

    print("① 矢量转栅格...")
    if os.path.exists(PolygonToRaster_shapes_dir) is False: os.mkdir(PolygonToRaster_shapes_dir)
    # 第一步，矢量转栅格
    for f in mask_shapes:
        arcpy.PolygonToRaster_conversion(mask_shapes_dir + f, 'FID',
                                         PolygonToRaster_shapes_dir + f.split('.')[0] + '1.tif')

    PolygonToRaster_tifs = []
    for f in os.listdir(PolygonToRaster_shapes_dir):
        if f.split('.')[1] == 'tif' and len(f.split('.')) == 2:
            PolygonToRaster_tifs.append(f)
    print("① 矢量转栅格 完成！")

    # 第二步，栅格计算
    print("② 栅格计算...")
    if os.path.exists(RasterCalculator_dir) is False: os.mkdir(RasterCalculator_dir)
    for f in PolygonToRaster_tifs:
        # print(f)
        out_A_Calculator = Con(PolygonToRaster_shapes_dir + f, GRIDCRO2D_temptif, 1)
        out_A_Calculator.save(RasterCalculator_dir + f.split('.')[0] + '2.tif')

    if os.path.exists(RasterReclass_dir) is False: os.mkdir(RasterReclass_dir)
    RasterReclass_tifs = []
    for f in os.listdir(RasterCalculator_dir):
        if f.split('.')[1] == 'tif' and len(f.split('.')) == 2:
            RasterReclass_tifs.append(f)
    print("② 栅格计算 完成！")

    # 第三步，重分类，这里没用arcpy，用的rasterio
    print("③ 重分类...")
    for f in RasterReclass_tifs:
        inRaster_dir = RasterCalculator_dir + f
        inRaster_scr = rasterio.open(inRaster_dir)
        inRaster = rasterio.open(inRaster_dir).read(1)
        output_path = RasterReclass_dir + f.split('.')[0] + '3.tif'
        reclassified_data = inRaster.copy()
        reclassified_data[inRaster != 1] = 0  # 不是mask的像元点为0
        # print(reclassified_data)
        with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=inRaster_scr.height,
                width=inRaster_scr.width,
                count=1,
                dtype=reclassified_data.dtype,
                crs=inRaster_scr.crs,
                transform=inRaster_scr.transform,
        ) as dst:
            dst.write(reclassified_data, 1)

    if os.path.exists(Mosaic_dir) is False: os.mkdir(Mosaic_dir)
    Mosaic_tifs = []
    for f in os.listdir(RasterReclass_dir):
        if f.split('.')[1] == 'tif' and len(f.split('.')) == 2:
            Mosaic_tifs.append(f)
    # print(Mosaic_tifs)
    print("③ 重分类 完成！")

    # 第四步，镶嵌
    print("④ 镶嵌...")
    for f in Mosaic_tifs:
        # print('copy ' + GRIDCRO2D_temptif + ' ' + Mosaic_dir + f + '')
        os.system('copy ' + GRIDCRO2D_temptif + ' ' + Mosaic_dir + f + '')
        mosaicraster = arcpy.Mosaic_management(RasterReclass_dir + f, Mosaic_dir + f)
    print("④ 镶嵌 完成！")

    # 第五步，重命名tiff为mask变量名
    print("⑤ 重命名...")
    if os.path.exists(Regins_tiff_dir) is False: os.mkdir(Regins_tiff_dir)
    masktifs = []
    for f in os.listdir(Mosaic_dir):
        if f.split('.')[1] == 'tif' and len(f.split('.')) == 2:
            # print(f)
            masktifs.append(f)
    for f in masktifs:
        # print('awdawdawd', f)
        for i in shapenames:
            if f.split('.')[0].replace('123', '') == i:
                # print(11)
                os.system('copy ' + Mosaic_dir + f + ' ' + Regins_tiff_dir + shapenames_var[
                    shapenames.index(i)] + '.tif' + '')  # 更名并保存到文件夹
    print("⑤ 重命名 完成！")

    # # 第六步，生成nc文件，目前无法解决python3.9来运行这部分会报错的问题，上面运行完后换回3.10运行ISAMMASK_generation


if __name__ == '__main__':
    namelist_WCAS = f90nml.read(f"namelist.WCAS")
    mask_shpfilenames = namelist_WCAS['ISAMcontrol']['mask_shpfilenames']
    mask_varnames = namelist_WCAS['ISAMcontrol']['mask_varnames']
    print(mask_shpfilenames,mask_varnames)
    if isinstance(mask_shpfilenames, list):
        mask_shpfilenames_l = []
        for i in mask_shpfilenames:
            mask_shpfilenames_l.append(i)
    else:
        mask_shpfilenames_l = []
        mask_shpfilenames_l.append(mask_shpfilenames)

    if isinstance(mask_varnames, list):
        mask_varnames_l = []
        for i in mask_varnames:
            mask_varnames_l.append(i)
    else:
        mask_varnames_l = []
        mask_varnames_l.append(mask_varnames)

    print(mask_shpfilenames_l,mask_varnames_l)

    ISAMmask_ArcpyRegionsProcess(
        Workdir="",  #
        mask_shapes_dir=r"maskshps/",  #
        GRIDCRO2D_temptif=r"temp.tif",  #
        shapenames=mask_shpfilenames_l,  #
        shapenames_var=mask_varnames_l,  #
    )

    os.system("echo "">ISAMmask_ArcpyRegionsProcess_forWCAS.OK")  # 创建MEIAT运行完成flag文件
