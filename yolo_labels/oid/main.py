from yolo_labels.oid.reader import OIDLabelReader, DatasetType
from yolo_labels.shared import LabelWriter
from yolo_labels.shared import ObjectNameId
import os


input_path = '/Users/bothmena/Projects/datasets/OID'
output_path = os.path.join(input_path, 'oid_to_yolo/output_train.txt')
label_id_mapper = {
    '/m/0199g': ObjectNameId.BICYCLE.value,
    '/m/0k4j': ObjectNameId.CAR.value,
    '/m/04_sv': ObjectNameId.MOTORCYCLE.value,
    '/m/076bq': ObjectNameId.SEGWAY.value,
    '/m/01jfm_': ObjectNameId.LICENCE_PLATE.value,
}

reader = OIDLabelReader(input_path=input_path,
                        dataset_type=DatasetType.TRAIN,
                        label_id_mapper=label_id_mapper,
                        unnormalize_bbox=False,
                        )
writer = LabelWriter(output_path='dist/output.txt', reader=reader, overwrite_existent=True)

writer.write_annotations()

# objects downloaded from OID
# /m/0199g,Bicycle
# /m/0k4j,Car
# /m/04_sv,Motorcycle
# /m/076bq,Segway
# /m/01jfm_,Vehicle registration plate

# Object that can be really interesting
# /m/0f6nr,Unicycle
# /m/03kt2w,Stationary bicycle
# /m/01bjv,Bus
# /m/0pg52,Taxi
# /m/07jdr,Train
# /m/01xs3r,Jet ski
# /m/03p3bw,Bicycle helmet
# /m/01g317,Person
# /m/01x3jk,Snowmobile
# /m/03bt1vf,Woman
# /m/05r655,Girl
# /m/0dzct,Human face
# /m/0dzf4,Human arm
# /m/0k0pj,Human nose
# /m/0k5j,Aircraft
# /m/0k65p,Human hand
# /m/0zvk5,Helmet
