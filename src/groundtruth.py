from pycocotools.coco import COCO

def getGroundTruth(file_path,cat_name):
    
    id_img = int(file_path[-16:-4])
    coco = COCO('src/instances_train2017.json')
    annotation_ids = coco.getAnnIds(id_img)
    annotations = coco.loadAnns(annotation_ids)
    for i in range(len(annotations)):
        entity_id = annotations[i]["category_id"]
        entity = coco.loadCats(entity_id)[0]["name"]
        if entity==cat_name:
            return True
    return False    
    
