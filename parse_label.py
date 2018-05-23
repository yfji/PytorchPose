import json

label_path='/mnt/sda6/Keypoint/ai_challenger_keypoint_train_20170909/keypoint_train_annotations_20170909.json'

dataset=json.load(open(label_path,'r'))

print(len(dataset))
entry=dataset[0]

print(entry.keys())
print(entry['url'])
print(entry['image_id'])

for entry in dataset:
    if len(entry['human_annotations'].keys())>3:

        print(entry['human_annotations'])
        print(entry['keypoint_annotations'])
        break