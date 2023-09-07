# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

from detectron2.data.datasets.coco import load_coco_json, load_sem_seg

EntityV2_panoptic_CATEGORIES = [
{'supercategory': 'person', 'id': 1, 'name': 'man', 'c_name': '男人', 'isthing': True},
{'supercategory': 'land_transportation', 'id': 2, 'name': 'car', 'c_name': '小汽车', 'isthing': True},
{'supercategory': 'inanimate_natural_objects', 'id': 3, 'name': 'sky', 'c_name': '天空', 'isthing': False},
{'supercategory': 'seat_furniture', 'id': 4, 'name': 'chair', 'c_name': '椅子', 'isthing': True},
{'supercategory': 'utility', 'id': 5, 'name': 'utility_ot', 'c_name': '其他工具', 'isthing': True},
{'supercategory': 'non_individual_plants', 'id': 6, 'name': 'tree', 'c_name': '树木', 'isthing': False},
{'supercategory': 'traffic_facility', 'id': 7, 'name': 'street_light', 'c_name': '路灯', 'isthing': True},
{'supercategory': 'individual_plants', 'id': 8, 'name': 'potted_plants', 'c_name': '盆栽植物', 'isthing': False},
{'supercategory': 'media', 'id': 9, 'name': 'signboard', 'c_name': '简介牌', 'isthing': True},
{'supercategory': 'facility', 'id': 10, 'name': 'facility_ot', 'c_name': '其他设施类', 'isthing': True},
{'supercategory': 'rigid_container', 'id': 11, 'name': 'rigid_container_ot', 'c_name': '其他定型容器', 'isthing': True},
{'supercategory': 'non_individual_plants', 'id': 12, 'name': 'grass', 'c_name': '草地', 'isthing': False},
{'supercategory': 'media', 'id': 13, 'name': 'sculpture', 'c_name': '雕塑雕像', 'isthing': True},
{'supercategory': 'media', 'id': 14, 'name': 'book', 'c_name': '书', 'isthing': True},
{'supercategory': 'non_individual_plants', 'id': 15, 'name': 'shrub', 'c_name': '灌木', 'isthing': False},
{'supercategory': 'rigid_container', 'id': 16, 'name': 'box', 'c_name': '盒子箱子', 'isthing': True},
{'supercategory': 'table', 'id': 17, 'name': 'table_ot', 'c_name': '其他桌子', 'isthing': True},
{'supercategory': 'non-building_houses', 'id': 18, 'name': 'electricpole', 'c_name': '电线杆', 'isthing': False},
{'supercategory': 'furniture', 'id': 19, 'name': 'noncommon_furniture', 'c_name': '非常用家具', 'isthing': True},
{'supercategory': 'light', 'id': 20, 'name': 'can_light', 'c_name': '筒灯', 'isthing': True},
{'supercategory': 'wall', 'id': 21, 'name': 'concretewall', 'c_name': '水泥墙', 'isthing': False},
{'supercategory': 'floor_structure', 'id': 22, 'name': 'floor_structure_ot', 'c_name': '其他地面结构', 'isthing': False},
{'supercategory': 'barrier', 'id': 23, 'name': 'sunk_fence', 'c_name': '矮墙', 'isthing': False},
{'supercategory': 'media', 'id': 24, 'name': 'painting', 'c_name': '绘画类', 'isthing': True},
{'supercategory': 'building_houses', 'id': 25, 'name': 'house', 'c_name': '房屋', 'isthing': False},
{'supercategory': 'traffic_facility', 'id': 26, 'name': 'traffic_sign', 'c_name': '交通标志', 'isthing': True},
{'supercategory': 'barrier', 'id': 27, 'name': 'barrier_ot', 'c_name': '其他障碍物', 'isthing': False},
{'supercategory': 'building_structure', 'id': 28, 'name': 'floor', 'c_name': '地板', 'isthing': False},
{'supercategory': 'media', 'id': 29, 'name': 'paper', 'c_name': '纸', 'isthing': True},
{'supercategory': 'underwater_vehicle', 'id': 30, 'name': 'boat', 'c_name': '小型船只', 'isthing': True},
{'supercategory': 'entertainment_appliances_ot', 'id': 31, 'name': 'entertainment_appliances_ot', 'c_name': '其他娱乐设施', 'isthing': True},
{'supercategory': 'rigid_container', 'id': 32, 'name': 'bottle', 'c_name': '瓶子', 'isthing': True},
{'supercategory': 'solid', 'id': 33, 'name': 'mountain', 'c_name': '山', 'isthing': False},
{'supercategory': 'birds', 'id': 34, 'name': 'birds_ot', 'c_name': '其他飞禽', 'isthing': True},
{'supercategory': 'common_furniture', 'id': 35, 'name': 'cushion', 'c_name': '抱枕', 'isthing': True},
{'supercategory': 'building_structure', 'id': 36, 'name': 'pole', 'c_name': '细杆', 'isthing': False},
{'supercategory': 'cup', 'id': 37, 'name': 'cup_ot', 'c_name': '其他杯子', 'isthing': True},
{'supercategory': 'bag', 'id': 38, 'name': 'backpack', 'c_name': '背包', 'isthing': True},
{'supercategory': 'solid', 'id': 39, 'name': 'soil', 'c_name': '泥土', 'isthing': False},
{'supercategory': 'barrier', 'id': 40, 'name': 'fence', 'c_name': '栅栏', 'isthing': False},
{'supercategory': 'non-building_houses', 'id': 41, 'name': 'non-buildinghouse_ot', 'c_name': '其他非建筑房屋', 'isthing': False},
{'supercategory': 'light', 'id': 42, 'name': 'sconce', 'c_name': '壁灯', 'isthing': True},
{'supercategory': 'fooddrink', 'id': 43, 'name': 'fooddrink_ot', 'c_name': '其他食物饮品', 'isthing': True},
{'supercategory': 'solid', 'id': 44, 'name': 'stone', 'c_name': '小石头', 'isthing': False},
{'supercategory': 'building_structure', 'id': 45, 'name': 'gutter', 'c_name': '排水沟', 'isthing': False},
{'supercategory': 'bag', 'id': 46, 'name': 'handbag', 'c_name': '手提包', 'isthing': True},
{'supercategory': 'common_furniture', 'id': 47, 'name': 'decoration', 'c_name': '装饰物', 'isthing': True},
{'supercategory': 'traffic_facility', 'id': 48, 'name': 'electric_wire', 'c_name': '电线电缆', 'isthing': True},
{'supercategory': 'clean_tool', 'id': 49, 'name': 'trash_bin', 'c_name': '垃圾桶', 'isthing': True},
{'supercategory': 'artifical_ground', 'id': 50, 'name': 'road', 'c_name': '马路', 'isthing': False},
{'supercategory': 'kitchen_tool', 'id': 51, 'name': 'plate', 'c_name': '盘子', 'isthing': True},
{'supercategory': 'drink', 'id': 52, 'name': 'drink_ot', 'c_name': '其他饮品', 'isthing': True},
{'supercategory': 'building_structure', 'id': 53, 'name': 'ceiling', 'c_name': '天花板', 'isthing': False},
{'supercategory': 'fluid', 'id': 54, 'name': 'sea', 'c_name': '海', 'isthing': False},
{'supercategory': 'toy', 'id': 55, 'name': 'toy_ot', 'c_name': '其他玩具', 'isthing': True},
{'supercategory': 'media', 'id': 56, 'name': 'pen', 'c_name': '笔', 'isthing': True},
{'supercategory': 'media', 'id': 57, 'name': 'flag', 'c_name': '旗帜', 'isthing': True},
{'supercategory': 'solid', 'id': 58, 'name': 'rock', 'c_name': '岩石', 'isthing': False},
{'supercategory': 'outdoor_supplies', 'id': 59, 'name': 'outdoor_supplies_ot', 'c_name': '其他户外休闲用品', 'isthing': True},
{'supercategory': 'common_furniture', 'id': 60, 'name': 'curtain', 'c_name': '窗帘', 'isthing': True},
{'supercategory': 'light', 'id': 61, 'name': 'chandilier', 'c_name': '吊灯', 'isthing': True},
{'supercategory': 'fish', 'id': 62, 'name': 'fish_ot', 'c_name': '其他鱼类', 'isthing': True},
{'supercategory': 'building_structure', 'id': 63, 'name': 'window', 'c_name': '窗户', 'isthing': False},
{'supercategory': 'light', 'id': 64, 'name': 'light_ot', 'c_name': '其他灯', 'isthing': True},
{'supercategory': 'entertainment_appliances', 'id': 65, 'name': 'plugs_and_sockets', 'c_name': '插头插座', 'isthing': True},
{'supercategory': 'outdoor_supplies', 'id': 66, 'name': 'rope', 'c_name': '绳子', 'isthing': True},
{'supercategory': 'building_houses', 'id': 67, 'name': 'building_houses_ot', 'c_name': '其他建筑房屋', 'isthing': False},
{'supercategory': 'land_transportation', 'id': 68, 'name': 'land_transportation_ot', 'c_name': '其他陆地交通工具', 'isthing': True},
{'supercategory': 'land_transportation', 'id': 69, 'name': 'bicycle', 'c_name': '自行车', 'isthing': True},
{'supercategory': 'seat_furniture', 'id': 70, 'name': 'stool', 'c_name': '凳子', 'isthing': True},
{'supercategory': 'barrier', 'id': 71, 'name': 'wiredfence', 'c_name': '铁丝网栅栏', 'isthing': False},
{'supercategory': 'kitchen_tool', 'id': 72, 'name': 'kitchen_tool_ot', 'c_name': '其他厨房工具', 'isthing': True},
{'supercategory': 'building_structure', 'id': 73, 'name': 'building_structure_ot', 'c_name': '非建筑结构', 'isthing': False},
{'supercategory': 'building_houses', 'id': 74, 'name': 'high-rise', 'c_name': '高楼', 'isthing': False},
{'supercategory': 'blanket_furniture', 'id': 75, 'name': 'mat', 'c_name': '垫子', 'isthing': True},
{'supercategory': 'seat_furniture', 'id': 76, 'name': 'bench', 'c_name': '长凳', 'isthing': True},
{'supercategory': 'non_individual_plants', 'id': 77, 'name': 'nonindividual_plants_ot', 'c_name': '其他非独立植物', 'isthing': False},
{'supercategory': 'artifical_ground', 'id': 78, 'name': 'sidewalk', 'c_name': '人行道', 'isthing': False},
{'supercategory': 'media', 'id': 79, 'name': 'billboard', 'c_name': '广告牌', 'isthing': True},
{'supercategory': 'plant_part', 'id': 80, 'name': 'flower', 'c_name': '花', 'isthing': True},
{'supercategory': 'rigid_container', 'id': 81, 'name': 'bowl', 'c_name': '碗', 'isthing': True},
{'supercategory': 'plant_part', 'id': 82, 'name': 'leaf', 'c_name': '树叶', 'isthing': True},
{'supercategory': 'cow', 'id': 83, 'name': 'cow_ot', 'c_name': '其他牛', 'isthing': True},
{'supercategory': 'dessert_snacks', 'id': 84, 'name': 'dessert_snacks_ot', 'c_name': '其他甜点小吃', 'isthing': True},
{'supercategory': 'mammal', 'id': 85, 'name': 'dog', 'c_name': '狗', 'isthing': True},
{'supercategory': 'bag', 'id': 86, 'name': 'bag_ot', 'c_name': '其他包', 'isthing': True},
{'supercategory': 'media', 'id': 87, 'name': 'photo', 'c_name': '照片', 'isthing': True},
{'supercategory': 'bath_tool', 'id': 88, 'name': 'bath_tool_ot', 'c_name': '其他沐浴工具', 'isthing': True},
{'supercategory': 'bath_tool', 'id': 89, 'name': 'towel', 'c_name': '毛巾', 'isthing': True},
{'supercategory': 'building_structure', 'id': 90, 'name': 'step', 'c_name': '台阶', 'isthing': False},
{'supercategory': 'light', 'id': 91, 'name': 'ceiling_lamp', 'c_name': '吸顶灯', 'isthing': True},
{'supercategory': 'sports_musical', 'id': 92, 'name': 'musical_instrument', 'c_name': '乐器', 'isthing': True},
{'supercategory': 'rigid_container', 'id': 93, 'name': 'barrel', 'c_name': '桶', 'isthing': True},
{'supercategory': 'traffic_facility', 'id': 94, 'name': 'traffic_light', 'c_name': '红绿灯', 'isthing': True},
{'supercategory': 'cup', 'id': 95, 'name': 'wineglass', 'c_name': '酒杯', 'isthing': True},
{'supercategory': 'flexible_container', 'id': 96, 'name': 'plastic_bag', 'c_name': '塑料袋', 'isthing': True},
{'supercategory': 'cloth', 'id': 97, 'name': 'cloth_ot', 'c_name': '其他服饰类', 'isthing': True},
{'supercategory': 'rigid_container', 'id': 98, 'name': 'jar', 'c_name': '玻璃罐', 'isthing': True},
{'supercategory': 'telephone', 'id': 99, 'name': 'mobilephone', 'c_name': '手机', 'isthing': True},
{'supercategory': 'building_structure', 'id': 100, 'name': 'pipe', 'c_name': '管道', 'isthing': False},
{'supercategory': 'entertainment_appliances', 'id': 101, 'name': 'cable', 'c_name': '连接线', 'isthing': True},
{'supercategory': 'fitness_equipment', 'id': 102, 'name': 'fitness_equipment_ot', 'c_name': '其他健身设备', 'isthing': True},
{'supercategory': 'media', 'id': 103, 'name': 'poster', 'c_name': '海报类', 'isthing': True},
{'supercategory': 'cup', 'id': 104, 'name': 'glass', 'c_name': '玻璃杯', 'isthing': True},
{'supercategory': 'plant_part', 'id': 105, 'name': 'branch', 'c_name': '树枝', 'isthing': True},
{'supercategory': 'repair_tool', 'id': 106, 'name': 'repair_tool_ot', 'c_name': '其他修理工具', 'isthing': True},
{'supercategory': 'beddings', 'id': 107, 'name': 'pillow', 'c_name': '枕头', 'isthing': True},
{'supercategory': 'cabinets', 'id': 108, 'name': 'cabinets_ot', 'c_name': '其他橱柜', 'isthing': True},
{'supercategory': 'fruit', 'id': 109, 'name': 'apple', 'c_name': '苹果', 'isthing': True},
{'supercategory': 'mammal', 'id': 110, 'name': 'sheep', 'c_name': '羊', 'isthing': True},
{'supercategory': 'fluid', 'id': 111, 'name': 'lake', 'c_name': '湖', 'isthing': False},
{'supercategory': 'doll', 'id': 112, 'name': 'doll_ot', 'c_name': '其他玩偶', 'isthing': True},
{'supercategory': 'fruit', 'id': 113, 'name': 'fruit_ot', 'c_name': '其他水果', 'isthing': True},
{'supercategory': 'solid', 'id': 114, 'name': 'sand', 'c_name': '沙子', 'isthing': False},
{'supercategory': 'cabinets', 'id': 115, 'name': 'kitchen cabinets', 'c_name': '厨房里的柜子', 'isthing': True},
{'supercategory': 'non-building_houses', 'id': 116, 'name': 'bridge', 'c_name': '桥', 'isthing': False},
{'supercategory': 'fluid', 'id': 117, 'name': 'river', 'c_name': '河', 'isthing': False},
{'supercategory': 'plant_part', 'id': 118, 'name': 'trunk', 'c_name': '树干', 'isthing': True},
{'supercategory': 'media', 'id': 119, 'name': 'media_ot', 'c_name': '其他传媒类', 'isthing': True},
{'supercategory': 'wall', 'id': 120, 'name': 'wall_ot', 'c_name': '其他墙', 'isthing': False},
{'supercategory': 'lab_tool', 'id': 121, 'name': 'candle', 'c_name': '蜡烛', 'isthing': True},
{'supercategory': 'dessert_snacks', 'id': 122, 'name': 'bread', 'c_name': '面包', 'isthing': True},
{'supercategory': 'birds', 'id': 123, 'name': 'duck', 'c_name': '鸭', 'isthing': True},
{'supercategory': 'birds', 'id': 124, 'name': 'pigeon', 'c_name': '鸽子', 'isthing': True},
{'supercategory': 'kitchen_tool', 'id': 125, 'name': 'spoon', 'c_name': '勺子', 'isthing': True},
{'supercategory': 'artifical_ground', 'id': 126, 'name': 'park_ground', 'c_name': '公园地面', 'isthing': False},
{'supercategory': 'artifical_ground', 'id': 127, 'name': 'artifical_ground_ot', 'c_name': '其他人造地面', 'isthing': False},
{'supercategory': 'fluid', 'id': 128, 'name': 'fluid_ot', 'c_name': '其他液体', 'isthing': False},
{'supercategory': 'table', 'id': 129, 'name': 'dining_table', 'c_name': '餐桌', 'isthing': True},
{'supercategory': 'vegetable', 'id': 130, 'name': 'pumkin', 'c_name': '南瓜', 'isthing': True},
{'supercategory': 'fluid', 'id': 131, 'name': 'snow', 'c_name': '雪', 'isthing': False},
{'supercategory': 'horse', 'id': 132, 'name': 'horse_ot', 'c_name': '其他马', 'isthing': True},
{'supercategory': 'vegetable', 'id': 133, 'name': 'vegetable_ot', 'c_name': '其他蔬菜', 'isthing': True},
{'supercategory': 'flexible_container', 'id': 134, 'name': 'flexible_container_ot', 'c_name': '其他可改变外形的容器', 'isthing': True},
{'supercategory': 'entertainment_appliances', 'id': 135, 'name': 'surveillance_camera', 'c_name': '监控器', 'isthing': True},
{'supercategory': 'land_transportation', 'id': 136, 'name': 'motorbike', 'c_name': '摩托车', 'isthing': True},
{'supercategory': 'sofa', 'id': 137, 'name': 'ordniary_sofa', 'c_name': '普通沙发', 'isthing': True},
{'supercategory': 'building_structure', 'id': 138, 'name': 'banister', 'c_name': '扶手', 'isthing': False},
{'supercategory': 'entertainment_appliances', 'id': 139, 'name': 'laptop', 'c_name': '笔记本电脑', 'isthing': True},
{'supercategory': 'outdoor_supplies', 'id': 140, 'name': 'umbrella', 'c_name': '雨伞', 'isthing': True},
{'supercategory': 'kitchen_tool', 'id': 141, 'name': 'faucet', 'c_name': '水龙头', 'isthing': True},
{'supercategory': 'mammal', 'id': 142, 'name': 'mammal_ot', 'c_name': '其他哺乳动物', 'isthing': True},
{'supercategory': 'building', 'id': 143, 'name': 'building_ot', 'c_name': '其他建筑类', 'isthing': False},
{'supercategory': 'clean_tool', 'id': 144, 'name': 'napkin', 'c_name': '餐巾', 'isthing': True},
{'supercategory': 'land_transportation', 'id': 145, 'name': 'bus', 'c_name': '公交车', 'isthing': True},
{'supercategory': 'entertainment_appliances', 'id': 146, 'name': 'speaker', 'c_name': '音响', 'isthing': True},
{'supercategory': 'entertainment_appliances', 'id': 147, 'name': 'camera', 'c_name': '照相机', 'isthing': True},
{'supercategory': 'common_furniture', 'id': 148, 'name': 'mirror', 'c_name': '镜子', 'isthing': True},
{'supercategory': 'boat_part', 'id': 149, 'name': 'paddle', 'c_name': '桨', 'isthing': True},
{'supercategory': 'dessert_snacks', 'id': 150, 'name': 'cake', 'c_name': '糕饼', 'isthing': True},
{'supercategory': 'footwear', 'id': 151, 'name': 'sneakers', 'c_name': '运动鞋', 'isthing': True},
{'supercategory': 'flexible_container', 'id': 152, 'name': 'basket', 'c_name': '篮子', 'isthing': True},
{'supercategory': 'building_structure', 'id': 153, 'name': 'ventilation', 'c_name': '排气孔', 'isthing': False},
{'supercategory': 'underwater_vehicle', 'id': 154, 'name': 'sailboat', 'c_name': '帆船', 'isthing': True},
{'supercategory': 'underwater_vehicle', 'id': 155, 'name': 'ship', 'c_name': '大轮船', 'isthing': True},
{'supercategory': 'flexible_container', 'id': 156, 'name': 'can', 'c_name': '易拉罐', 'isthing': True},
{'supercategory': 'mammal', 'id': 157, 'name': 'cat', 'c_name': '猫', 'isthing': True},
{'supercategory': 'entertainment_appliances', 'id': 158, 'name': 'screen', 'c_name': '显示器', 'isthing': True},
{'supercategory': 'drink', 'id': 159, 'name': 'wine', 'c_name': '葡萄酒', 'isthing': True},
{'supercategory': 'fruit', 'id': 160, 'name': 'orange', 'c_name': '橘子', 'isthing': True},
{'supercategory': 'bed', 'id': 161, 'name': 'bedroom bed', 'c_name': '卧室床', 'isthing': True},
{'supercategory': 'ball', 'id': 162, 'name': 'ball_ot', 'c_name': '其他球类', 'isthing': True},
{'supercategory': 'non_individual_plants', 'id': 163, 'name': 'log', 'c_name': '原木', 'isthing': False},
{'supercategory': 'entertainment_appliances', 'id': 164, 'name': 'switch', 'c_name': '开关', 'isthing': True},
{'supercategory': 'mammal', 'id': 165, 'name': 'elephant', 'c_name': '大象', 'isthing': True},
{'supercategory': 'blanket_furniture', 'id': 166, 'name': 'blanket', 'c_name': '毛毯', 'isthing': True},
{'supercategory': 'air_vehicle', 'id': 167, 'name': 'airplane', 'c_name': '飞机', 'isthing': True},
{'supercategory': 'energyfacility', 'id': 168, 'name': 'kiosk', 'c_name': '电话亭', 'isthing': True},
{'supercategory': 'entertainment_appliances', 'id': 169, 'name': 'television', 'c_name': '电视机', 'isthing': True},
{'supercategory': 'land_transportation', 'id': 170, 'name': 'trolley', 'c_name': '手推车', 'isthing': True},
{'supercategory': 'insect', 'id': 171, 'name': 'bee', 'c_name': '蜜蜂', 'isthing': True},
{'supercategory': 'solid', 'id': 172, 'name': 'gravel', 'c_name': '砂砾', 'isthing': False},
{'supercategory': 'sofa', 'id': 173, 'name': 'couch', 'c_name': '长沙发', 'isthing': True},
{'supercategory': 'land_transportation', 'id': 174, 'name': 'van', 'c_name': '厢式货车', 'isthing': True},
{'supercategory': 'meat', 'id': 175, 'name': 'meat_ot', 'c_name': '其他肉', 'isthing': True},
{'supercategory': 'accessories', 'id': 176, 'name': 'accessories_ot', 'c_name': '其他服饰类', 'isthing': True},
{'supercategory': 'blanket_furniture', 'id': 177, 'name': 'blanket_furniture_ot', 'c_name': '其他毯子', 'isthing': True},
{'supercategory': 'common_furniture', 'id': 178, 'name': 'hanger', 'c_name': '衣架', 'isthing': True},
{'supercategory': 'blanket_furniture', 'id': 179, 'name': 'rug', 'c_name': '地毯', 'isthing': True},
{'supercategory': 'flexible_container', 'id': 180, 'name': 'paper_bag', 'c_name': '纸袋', 'isthing': True},
{'supercategory': 'entertainment_appliances', 'id': 181, 'name': 'remote_control', 'c_name': '遥控器', 'isthing': True},
{'supercategory': 'kitchen_tool', 'id': 182, 'name': 'kitchen_sink', 'c_name': '盥洗', 'isthing': True},
{'supercategory': 'kitchen_tool', 'id': 183, 'name': 'fork', 'c_name': '叉', 'isthing': True},
{'supercategory': 'kitchen_pot', 'id': 184, 'name': 'kitchen_pot_ot', 'c_name': '其他厨房用锅', 'isthing': True},
{'supercategory': 'insect', 'id': 185, 'name': 'insect_ot', 'c_name': '其他昆虫类', 'isthing': True},
{'supercategory': 'solid', 'id': 186, 'name': 'dirt', 'c_name': '贫瘠土地', 'isthing': False},
{'supercategory': 'artifical_ground', 'id': 187, 'name': 'path', 'c_name': '小径', 'isthing': False},
{'supercategory': 'underwater_vehicle', 'id': 188, 'name': 'underwater_vehicle_ot', 'c_name': '其他水中交通工具', 'isthing': True},
{'supercategory': 'sofa', 'id': 189, 'name': 'double_sofa', 'c_name': '双人沙发', 'isthing': True},
{'supercategory': 'seasoning', 'id': 190, 'name': 'condiment', 'c_name': '调味品', 'isthing': True},
{'supercategory': 'fooddrink', 'id': 191, 'name': 'drug', 'c_name': '药品', 'isthing': True},
{'supercategory': 'knife', 'id': 192, 'name': 'table-knife', 'c_name': '餐刀', 'isthing': True},
{'supercategory': 'fitness_equipment', 'id': 193, 'name': 'gym_equipment', 'c_name': '室内健身器材', 'isthing': True},
{'supercategory': 'footwear', 'id': 194, 'name': 'hose', 'c_name': '靴子', 'isthing': True},
{'supercategory': 'fruit', 'id': 195, 'name': 'peach', 'c_name': '桃', 'isthing': True},
{'supercategory': 'rigid_container', 'id': 196, 'name': 'flowerpot', 'c_name': '花盆', 'isthing': True},
{'supercategory': 'toy', 'id': 197, 'name': 'ballon', 'c_name': '气球', 'isthing': True},
{'supercategory': 'dessert_snacks', 'id': 198, 'name': 'bagel', 'c_name': '硬面包', 'isthing': True},
{'supercategory': 'non-building_houses', 'id': 199, 'name': 'tent', 'c_name': '帐篷', 'isthing': False},
{'supercategory': 'entertainment_appliances', 'id': 200, 'name': 'tv_receiver', 'c_name': '电视接收器', 'isthing': True},
{'supercategory': 'cabinets', 'id': 201, 'name': 'nightstand', 'c_name': '床头柜', 'isthing': True},
{'supercategory': 'kitchen_appliances', 'id': 202, 'name': 'kitchen_appliances_ot', 'c_name': '其他厨房电器', 'isthing': True},
{'supercategory': 'fitness_equipment', 'id': 203, 'name': 'ski_pole', 'c_name': '滑雪杆', 'isthing': True},
{'supercategory': 'upper_body_clothing', 'id': 204, 'name': 'coat', 'c_name': '外套', 'isthing': True},
{'supercategory': 'kitchen_tool', 'id': 205, 'name': 'cutting_board', 'c_name': '砧板', 'isthing': True},
{'supercategory': 'building_structure', 'id': 206, 'name': 'stair', 'c_name': '楼梯', 'isthing': False},
{'supercategory': 'plant_part', 'id': 207, 'name': 'plant_part_ot', 'c_name': '其他植物部分', 'isthing': True},
{'supercategory': 'footwear', 'id': 208, 'name': 'footwear_ot', 'c_name': '其他鞋', 'isthing': True},
{'supercategory': 'entertainment_appliances', 'id': 209, 'name': 'microphone', 'c_name': '麦克风', 'isthing': True},
{'supercategory': 'fruit', 'id': 210, 'name': 'lemon', 'c_name': '柠檬', 'isthing': True},
{'supercategory': 'non_individual_plants', 'id': 211, 'name': 'vine', 'c_name': '藤蔓', 'isthing': False},
{'supercategory': 'upper_body_clothing', 'id': 212, 'name': 'upper_body_clothing_os', 'c_name': '其他上身类', 'isthing': True},
{'supercategory': 'hat', 'id': 213, 'name': 'hat_ot', 'c_name': '其他帽', 'isthing': True},
{'supercategory': 'vegetable', 'id': 214, 'name': 'mushroom', 'c_name': '蘑菇', 'isthing': True},
{'supercategory': 'vehicle_part', 'id': 215, 'name': 'tire', 'c_name': '轮胎', 'isthing': True},
{'supercategory': 'cabinets', 'id': 216, 'name': 'filling cabinets', 'c_name': '文件柜', 'isthing': True},
{'supercategory': 'ball', 'id': 217, 'name': 'billards', 'c_name': '台球', 'isthing': True},
{'supercategory': 'kitchen_tool', 'id': 218, 'name': 'teapot', 'c_name': '茶壶', 'isthing': True},
{'supercategory': 'building_structure', 'id': 219, 'name': 'awning', 'c_name': '遮篷', 'isthing': False},
{'supercategory': 'entertainment_appliances', 'id': 220, 'name': 'keyboard', 'c_name': '键盘', 'isthing': True},
{'supercategory': 'repair_tool', 'id': 221, 'name': 'ladder', 'c_name': '梯子', 'isthing': True},
{'supercategory': 'entertainment_appliances', 'id': 222, 'name': 'mouse', 'c_name': '鼠标', 'isthing': True},
{'supercategory': 'kitchen_tool', 'id': 223, 'name': 'platter', 'c_name': '大浅盘', 'isthing': True},
{'supercategory': 'seat_furniture', 'id': 224, 'name': 'seat_furniture_ot', 'c_name': '其他椅子', 'isthing': True},
{'supercategory': 'kitchen_tool', 'id': 225, 'name': 'spatula', 'c_name': '厨房用铲', 'isthing': True},
{'supercategory': 'toy', 'id': 226, 'name': 'kite', 'c_name': '风筝', 'isthing': True},
{'supercategory': 'light', 'id': 227, 'name': 'lantern', 'c_name': '灯笼', 'isthing': True},
{'supercategory': 'land_transportation', 'id': 228, 'name': 'train', 'c_name': '火车', 'isthing': True},
{'supercategory': 'sofa', 'id': 229, 'name': 'sofa_ot', 'c_name': '其他沙发', 'isthing': True},
{'supercategory': 'artifical_ground', 'id': 230, 'name': 'trail', 'c_name': '小道', 'isthing': False},
{'supercategory': 'table', 'id': 231, 'name': 'integrated_table_and_chair', 'c_name': '一体化桌椅', 'isthing': True},
{'supercategory': 'building_houses', 'id': 232, 'name': 'castle', 'c_name': '城堡', 'isthing': False},
{'supercategory': 'lab_tool', 'id': 233, 'name': 'lab_tool_ot', 'c_name': '其他实验工具', 'isthing': True},
{'supercategory': 'seasoning', 'id': 234, 'name': 'seasoning_ot', 'c_name': '其他调味品', 'isthing': True},
{'supercategory': 'land_transportation', 'id': 235, 'name': 'cable_car', 'c_name': '缆车', 'isthing': True},
{'supercategory': 'fast_food', 'id': 236, 'name': 'fast_food_ot', 'c_name': '其他快餐食品', 'isthing': True},
{'supercategory': 'artifical_ground', 'id': 237, 'name': 'railroad', 'c_name': '铁路轨道', 'isthing': False},
{'supercategory': 'kitchen_tool', 'id': 238, 'name': 'tableware', 'c_name': '餐具', 'isthing': True},
{'supercategory': 'artifical_ground', 'id': 239, 'name': 'court', 'c_name': '球场', 'isthing': False},
{'supercategory': 'toy', 'id': 240, 'name': 'played_blocks', 'c_name': '积木', 'isthing': True},
{'supercategory': 'light', 'id': 241, 'name': 'lightbulb', 'c_name': '灯泡', 'isthing': True},
{'supercategory': 'toy', 'id': 242, 'name': 'chess', 'c_name': '棋类', 'isthing': True},
{'supercategory': 'non_individual_plants', 'id': 243, 'name': 'crops', 'c_name': '庄稼', 'isthing': False},
{'supercategory': 'bag', 'id': 244, 'name': 'suitcase', 'c_name': '手提箱', 'isthing': True},
{'supercategory': 'reptile', 'id': 245, 'name': 'lizard', 'c_name': '蜥蜴', 'isthing': True},
{'supercategory': 'rigid_container', 'id': 246, 'name': 'earthware_pot_with_handle', 'c_name': '带柄的陶罐', 'isthing': True},
{'supercategory': 'fast_food', 'id': 247, 'name': 'pizza', 'c_name': '披萨', 'isthing': True},
{'supercategory': 'building_houses', 'id': 248, 'name': 'tower', 'c_name': '塔', 'isthing': False},
{'supercategory': 'truck', 'id': 249, 'name': 'truck_ot', 'c_name': '其他卡车', 'isthing': True},
{'supercategory': 'fooddrink', 'id': 250, 'name': 'cigarette', 'c_name': '香烟', 'isthing': True},
{'supercategory': 'human_accessories', 'id': 251, 'name': 'scarf', 'c_name': '围巾', 'isthing': True},
{'supercategory': 'kitchen_appliances', 'id': 252, 'name': 'refrigerator', 'c_name': '冰箱', 'isthing': True},
{'supercategory': 'racket', 'id': 253, 'name': 'racket_ot', 'c_name': '其他拍子', 'isthing': True},
{'supercategory': 'deer', 'id': 254, 'name': 'giraffe', 'c_name': '长颈鹿', 'isthing': True},
{'supercategory': 'land_transportation', 'id': 255, 'name': 'scooter', 'c_name': '滑板车', 'isthing': True},
{'supercategory': 'appliances', 'id': 256, 'name': 'appliances_ot', 'c_name': '其他家电', 'isthing': True},
{'supercategory': 'fruit', 'id': 257, 'name': 'banana', 'c_name': '香蕉', 'isthing': True},
{'supercategory': 'fitness_equipment', 'id': 258, 'name': 'ski_board', 'c_name': '滑雪板', 'isthing': True},
{'supercategory': 'clean_tool', 'id': 259, 'name': 'clean_tool_ot', 'c_name': '其他清洁工具', 'isthing': True},
{'supercategory': 'toy', 'id': 260, 'name': 'dice', 'c_name': '筛子', 'isthing': True},
{'supercategory': 'dessert_snacks', 'id': 261, 'name': 'crumb', 'c_name': '面包屑', 'isthing': True},
{'supercategory': 'fruit', 'id': 262, 'name': 'commonfig', 'c_name': '无花果', 'isthing': True},
{'supercategory': 'vegetable', 'id': 263, 'name': 'tomato', 'c_name': '番茄', 'isthing': True},
{'supercategory': 'birds', 'id': 264, 'name': 'goose', 'c_name': '鹅', 'isthing': True},
{'supercategory': 'table', 'id': 265, 'name': 'desk', 'c_name': '书桌', 'isthing': True},
{'supercategory': 'media', 'id': 266, 'name': 'packaging_paper', 'c_name': '包装纸', 'isthing': True},
{'supercategory': 'toy', 'id': 267, 'name': 'poker', 'c_name': '扑克', 'isthing': True},
{'supercategory': 'non-building_houses', 'id': 268, 'name': 'pier', 'c_name': '码头', 'isthing': False},
{'supercategory': 'non-building_houses', 'id': 269, 'name': 'swimmingpool', 'c_name': '游泳池', 'isthing': False},
{'supercategory': 'fitness_equipment', 'id': 270, 'name': 'surfboard', 'c_name': '冲浪板', 'isthing': True},
{'supercategory': 'lab_tool', 'id': 271, 'name': ' medical_equipment', 'c_name': '医疗器械', 'isthing': True},
{'supercategory': 'footwear', 'id': 272, 'name': 'leather_shoes', 'c_name': '皮鞋', 'isthing': True},
{'supercategory': 'media', 'id': 273, 'name': 'blackboard', 'c_name': '黑板白板', 'isthing': True},
{'supercategory': 'vegetable', 'id': 274, 'name': 'egg', 'c_name': '鸡蛋', 'isthing': True},
{'supercategory': 'ventilation_appliances', 'id': 275, 'name': 'air_conditioner', 'c_name': '空调', 'isthing': True},
{'supercategory': 'entertainment_appliances', 'id': 276, 'name': 'gamepad', 'c_name': '游戏手柄', 'isthing': True},
{'supercategory': 'clean_tool', 'id': 277, 'name': 'pipe', 'c_name': '水管', 'isthing': True},
{'supercategory': 'insect', 'id': 278, 'name': 'moth', 'c_name': '飞蛾', 'isthing': True},
{'supercategory': 'crustacea', 'id': 279, 'name': 'crustacea_ot', 'c_name': '其他甲壳类动物', 'isthing': True},
{'supercategory': 'mollusca', 'id': 280, 'name': 'jellyfish', 'c_name': '水母', 'isthing': True},
{'supercategory': 'table', 'id': 281, 'name': 'table_cloth', 'c_name': '桌布', 'isthing': True},
{'supercategory': 'cabinets', 'id': 282, 'name': 'wardrobe', 'c_name': '衣柜', 'isthing': True},
{'supercategory': 'building_structure', 'id': 283, 'name': 'bar', 'c_name': '吧台', 'isthing': False},
{'supercategory': 'underwater_vehicle', 'id': 284, 'name': 'canoe', 'c_name': '皮划艇', 'isthing': True},
{'supercategory': 'birds', 'id': 285, 'name': 'chicken', 'c_name': '鸡', 'isthing': True},
{'supercategory': 'fluid', 'id': 286, 'name': 'pond', 'c_name': '池塘', 'isthing': False},
{'supercategory': 'upper_body_clothing', 'id': 287, 'name': 'T-shirt', 'c_name': 'T恤', 'isthing': True},
{'supercategory': 'footwear', 'id': 288, 'name': 'splippers', 'c_name': '拖鞋', 'isthing': True},
{'supercategory': 'media', 'id': 289, 'name': 'bulletin_board', 'c_name': '电子公告牌', 'isthing': True},
{'supercategory': 'fruit', 'id': 290, 'name': 'strawberry', 'c_name': '草莓', 'isthing': True},
{'supercategory': 'telephone', 'id': 291, 'name': 'corded_telephone', 'c_name': '有线电话', 'isthing': True},
{'supercategory': 'kitchen_tool', 'id': 292, 'name': 'saucer', 'c_name': '茶碟', 'isthing': True},
{'supercategory': 'reptile', 'id': 293, 'name': 'turtle', 'c_name': '龟', 'isthing': True},
{'supercategory': 'cup', 'id': 294, 'name': 'coffee cup', 'c_name': '咖啡杯', 'isthing': True},
{'supercategory': 'solid', 'id': 295, 'name': 'solid_ot', 'c_name': '其他固体', 'isthing': False},
{'supercategory': 'wall', 'id': 296, 'name': 'glasswall', 'c_name': '玻璃墙', 'isthing': False},
{'supercategory': 'vegetable', 'id': 297, 'name': 'carrot', 'c_name': '胡萝卜', 'isthing': True},
{'supercategory': 'clean_tool', 'id': 298, 'name': 'tissue', 'c_name': '抽纸', 'isthing': True},
{'supercategory': 'individual_plants', 'id': 299, 'name': 'individual_plants_ot', 'c_name': '其他独立植物', 'isthing': False},
{'supercategory': 'traffic_facility', 'id': 300, 'name': 'fire_hyrant', 'c_name': '消防龙头', 'isthing': True},
{'supercategory': 'footwear', 'id': 301, 'name': ' high-heeled_shoes', 'c_name': '高跟鞋', 'isthing': True},
{'supercategory': 'vegetable', 'id': 302, 'name': 'potato', 'c_name': '马铃薯', 'isthing': True},
{'supercategory': 'entertainment_appliances', 'id': 303, 'name': 'radiator', 'c_name': '暖气', 'isthing': True},
{'supercategory': 'repair_tool', 'id': 304, 'name': 'wrench', 'c_name': '扳手', 'isthing': True},
{'supercategory': 'stove', 'id': 305, 'name': 'gas_stove', 'c_name': '燃气炉', 'isthing': True},
{'supercategory': 'horse', 'id': 306, 'name': 'zebra', 'c_name': '斑马', 'isthing': True},
{'supercategory': 'non-building_houses', 'id': 307, 'name': 'fountain_ground', 'c_name': '喷泉台', 'isthing': False},
{'supercategory': 'knife', 'id': 308, 'name': 'knife_ot', 'c_name': '其他刀', 'isthing': True},
{'supercategory': 'weapon', 'id': 309, 'name': 'rifle', 'c_name': '步枪', 'isthing': True},
{'supercategory': 'mammal', 'id': 310, 'name': 'monkey', 'c_name': '猴子', 'isthing': True},
{'supercategory': 'individual_plants', 'id': 311, 'name': 'straw', 'c_name': '稻草', 'isthing': False},
{'supercategory': 'ball', 'id': 312, 'name': 'golf', 'c_name': '高尔夫球', 'isthing': True},
{'supercategory': 'lab_tool', 'id': 313, 'name': ' folder', 'c_name': '文件夹', 'isthing': True},
{'supercategory': 'human_accessories', 'id': 314, 'name': 'gloves', 'c_name': '手套', 'isthing': True},
{'supercategory': 'repair_tool', 'id': 315, 'name': 'screwdriver', 'c_name': '螺丝刀', 'isthing': True},
{'supercategory': 'kitchen_tool', 'id': 316, 'name': 'drinking_straw', 'c_name': '吸管', 'isthing': True},
{'supercategory': 'mammal', 'id': 317, 'name': 'pig', 'c_name': '猪', 'isthing': True},
{'supercategory': 'plant', 'id': 318, 'name': 'plant_ot', 'c_name': '其他植物', 'isthing': True},
{'supercategory': 'cabinets', 'id': 319, 'name': 'bathroom cabinets', 'c_name': '浴室柜', 'isthing': True},
{'supercategory': 'ventilation_appliances', 'id': 320, 'name': 'vent', 'c_name': '通风孔', 'isthing': True},
{'supercategory': 'clean_appliances', 'id': 321, 'name': 'washing_machine', 'c_name': '洗衣机', 'isthing': True},
{'supercategory': 'racket', 'id': 322, 'name': 'tennis_racket', 'c_name': '网球拍', 'isthing': True},
{'supercategory': 'kitchen_tool', 'id': 323, 'name': 'chopsticks', 'c_name': '筷子', 'isthing': True},
{'supercategory': 'mammal', 'id': 324, 'name': 'seal', 'c_name': '海豹', 'isthing': True},
{'supercategory': 'building_houses', 'id': 325, 'name': 'lighthouse', 'c_name': '灯塔', 'isthing': False},
{'supercategory': 'kitchen_appliances', 'id': 326, 'name': 'kettle', 'c_name': '水壶', 'isthing': True},
{'supercategory': 'fitness_equipment', 'id': 327, 'name': 'parachute', 'c_name': '降落伞', 'isthing': True},
{'supercategory': 'beddings', 'id': 328, 'name': 'blanket', 'c_name': '被子', 'isthing': True},
{'supercategory': 'drink', 'id': 329, 'name': 'juice', 'c_name': '果汁', 'isthing': True},
{'supercategory': 'kitchen_tool', 'id': 330, 'name': 'food_processor', 'c_name': '食品加工机', 'isthing': True},
{'supercategory': 'truck', 'id': 331, 'name': 'pickup_truck', 'c_name': '皮卡车', 'isthing': True},
{'supercategory': 'doll', 'id': 332, 'name': 'teddy_bear', 'c_name': '泰迪熊', 'isthing': True},
{'supercategory': 'deer', 'id': 333, 'name': 'deer_ot', 'c_name': '其他鹿', 'isthing': True},
{'supercategory': 'clock_furniture', 'id': 335, 'name': 'clock', 'c_name': '时钟', 'isthing': True},
{'supercategory': 'beddings', 'id': 336, 'name': 'beddings_ot', 'c_name': '其他床上用品', 'isthing': True},
{'supercategory': 'lab_tool', 'id': 337, 'name': 'tube', 'c_name': '试管', 'isthing': True},
{'supercategory': 'fluid', 'id': 338, 'name': 'fountain', 'c_name': '喷泉', 'isthing': False},
{'supercategory': 'non-building_houses', 'id': 339, 'name': 'parterre', 'c_name': '花坛', 'isthing': False},
{'supercategory': 'human_accessories', 'id': 340, 'name': 'human_accessories_ot', 'c_name': '其他人物服饰类', 'isthing': True},
{'supercategory': 'birds', 'id': 341, 'name': 'parrot', 'c_name': '鹦鹉', 'isthing': True},
{'supercategory': 'common_furniture', 'id': 342, 'name': 'toilet', 'c_name': '马桶', 'isthing': True},
{'supercategory': 'fitness_equipment', 'id': 343, 'name': 'dumbbell', 'c_name': '哑铃', 'isthing': True},
{'supercategory': 'fruit', 'id': 344, 'name': 'pear', 'c_name': '梨', 'isthing': True},
{'supercategory': 'fruit', 'id': 345, 'name': 'pineapple', 'c_name': '菠萝', 'isthing': True},
{'supercategory': 'building_houses', 'id': 346, 'name': 'temple', 'c_name': '寺庙', 'isthing': False},
{'supercategory': 'camel', 'id': 350, 'name': 'camel_ot', 'c_name': '其他骆驼', 'isthing': True},
{'supercategory': 'person', 'id': 640, 'name': 'person_ot', 'c_name': '无法分辨年龄性别的人', 'isthing': True},
{'supercategory': 'person', 'id': 641, 'name': 'woman', 'c_name': '女人', 'isthing': True},
{'supercategory': 'person', 'id': 642, 'name': 'boy', 'c_name': '男孩', 'isthing': True},
{'supercategory': 'person', 'id': 643, 'name': 'girl', 'c_name': '女孩', 'isthing': True},
]

def _get_entityv2_panoptic_meta():
    meta = {}

    ## thing_dataset_id and stuff_dataset_id share the same contiguous id
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}
    for i, cat in enumerate(EntityV2_panoptic_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        else:
            stuff_dataset_id_to_contiguous_id[cat["id"]] = i
    
    ## add new map for PanopticFPN Dataloader
    thing_contiguous_id_to_new_contiguous =  {}
    stuff_contiguous_id_to_new_contiguous =  {}
    for id_, (thing_id, contiguous_id) in enumerate(thing_dataset_id_to_contiguous_id.items()):
        thing_contiguous_id_to_new_contiguous[contiguous_id] = id_
    
    for id_, (stuff_id, contiguous_id) in enumerate(stuff_dataset_id_to_contiguous_id.items()):
        stuff_contiguous_id_to_new_contiguous[contiguous_id] = id_+1
    
    thing_classes = [k["name"] for k in EntityV2_panoptic_CATEGORIES]
    stuff_classes = [k["name"] for k in EntityV2_panoptic_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["stuff_classes"] = stuff_classes

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id
    meta["thing_contiguous_id_to_new_contiguous"] = thing_contiguous_id_to_new_contiguous
    meta["stuff_contiguous_id_to_new_contiguous"] = stuff_contiguous_id_to_new_contiguous

    return meta


def load_entityv2_panoptic_json(json_file, image_dir, gt_dir, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    ret = []
    for ann in json_info["annotations"]:
        image_id = int(ann["image_id"])
        image_file = os.path.join(image_dir, ann["file_name"])
        label_file = os.path.join(gt_dir, ann["file_name"].split(".")[0]+".png")
        # segments_info = ann["segments_info"]
        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "segments_info": segments_info,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    return ret

def register_entityv2_panoptic(name, metadata, image_root, panoptic_root, panoptic_json):
    """
    """
    panoptic_name = name
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    panoptic_json = os.path.join(_root, panoptic_json)
    panoptic_root = os.path.join(_root, panoptic_root)
    image_root = os.path.join(_root, image_root)
    DatasetCatalog.register(
        panoptic_name,
        lambda: load_entityv2_panoptic_json(panoptic_json, image_root, panoptic_root, metadata),
    )
    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )

_PREDEFINED_SPLITS = {
    "entityv2_panoptic_train": (
        "entityseg/images/entity_01_11580",
        "entityseg/annotations/panoptic_segmentation/panoptic_maps_train",
        "entityseg/annotations/panoptic_segmentation/entityv2_01_panoptic_train.json",
    ),
    "entityv2_panoptic_val": (
        "entityseg/images/entity_01_11580",
        "entityseg/annotations/panoptic_segmentation/panoptic_maps_val",
        "entityseg/annotations/panoptic_segmentation/entityv2_01_panoptic_val.json",
    ),
}

for name, (image_root, gt_root, json_path) in _PREDEFINED_SPLITS.items():
    metadata = _get_entityv2_panoptic_meta()
    register_entityv2_panoptic(name, metadata, image_root, gt_root, json_path)


