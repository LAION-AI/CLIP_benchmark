from subprocess import call
import os, json

GITHUB_MAIN_ORIGINAL_ANNOTATION_PATH = 'https://github.com/mehdidc/retrieval_annotations/releases/download/1.0.0/coco_{}_karpathy.json'
GITHUB_MAIN_PATH = 'https://raw.githubusercontent.com/adobe-research/Cross-lingual-Test-Dataset-XTD10/main/XTD10/'
SUPPORTED_LANGUAGES = ['es', 'it', 'ko', 'pl', 'ru', 'tr', 'zh']

IMAGE_INDEX_FILE = 'mscoco-multilingual_index.json'
IMAGE_INDEX_FILE_DOWNLOAD_NAME = 'test_image_names.txt'

CAPTIONS_FILE_DOWNLOAD_NAME = 'test_1kcaptions_{}.txt'
CAPTIONS_FILE_NAME = 'multilingual_test_1kcaptions_{}.json'

ORIGINAL_ANNOTATION_FILE_NAME = 'coco_{}_karpathy.json'


def _get_downloadable_file(filename, download_url, is_json=True):
    if (os.path.exists(filename) == False):
        print("Downloading", download_url)
        call("wget {} -O {}".format(download_url, filename), shell=True)
    with open(filename, 'r') as fp:
        if (is_json):
            return json.load(fp)
        return [line.strip() for line in fp.readlines()]


def create_english_annotation_file(root):
    print("Downloading multilingual_ms_coco index file")
    download_path = os.path.join(GITHUB_MAIN_PATH, IMAGE_INDEX_FILE_DOWNLOAD_NAME)
    target_images = _get_downloadable_file("multilingual_coco_temp_images.txt", download_path, False)

    print("Downloading multilingual_ms_coco english captions")
    download_path = os.path.join(GITHUB_MAIN_PATH, CAPTIONS_FILE_DOWNLOAD_NAME.format('en'))
    target_captions = _get_downloadable_file("multilingual_coco_temp_en_captions.txt", download_path, False)

    # Load original annotation files to match format
    train_ann = _get_downloadable_file(ORIGINAL_ANNOTATION_FILE_NAME.format('train'),
                                       GITHUB_MAIN_ORIGINAL_ANNOTATION_PATH.format('train'))
    val_ann = _get_downloadable_file(ORIGINAL_ANNOTATION_FILE_NAME.format('val'),
                                     GITHUB_MAIN_ORIGINAL_ANNOTATION_PATH.format('val'))
    test_ann = _get_downloadable_file(ORIGINAL_ANNOTATION_FILE_NAME.format('test'),
                                      GITHUB_MAIN_ORIGINAL_ANNOTATION_PATH.format('test'))

    def parse_original_annotation_file(data):
        image_2_image_sections = {}
        image_2_captions_sections = {}

        id_2_annotation_section = {d['image_id']: d for d in data['annotations']}

        for d in data['images']:
            file_name = d['file_name']
            image_2_image_sections[file_name] = d
            file_id = d['id']
            image_2_captions_sections[file_name] = id_2_annotation_section[file_id]

        return image_2_image_sections, image_2_captions_sections

    val_img_2_img_sections, val_img_2_captions_sections = parse_original_annotation_file(val_ann)
    test_img_2_img_sections, test_img_2_captions_sections = parse_original_annotation_file(test_ann)
    train_img_2_img_sections, train_img_2_captions_sections = parse_original_annotation_file(train_ann)

    all_keys, img_2_split = {}, {}
    for img_sections, captions_sections in [(val_img_2_img_sections, val_img_2_captions_sections),
                                            (test_img_2_img_sections, test_img_2_captions_sections),
                                            (train_img_2_img_sections, train_img_2_captions_sections)]:
        for k in img_sections.keys():
            all_keys[k] = 1
            img_2_split[k] = (img_sections, captions_sections)

    number_of_missing_files = 0
    new_image_sections, new_annotation_sections = [], []
    for new_img, new_txt in zip(target_images, target_captions):
        if (new_img not in all_keys):
            print("Missing file", new_img)
            number_of_missing_files += 1
            continue


        target_img_sections, target_captions_sections = img_2_split[new_img]
        img_section, captions_section = target_img_sections[new_img], target_captions_sections[new_img]

        # Create a new file name that includes the root split
        root_split = 'val2014' if 'val' in new_img else 'train2014'
        filename_with_root_split = "{}/{}".format(root_split,new_img)
        img_section['file_name'] = filename_with_root_split

        new_annotation_sections.append(captions_section)
        new_image_sections.append(img_section)
    if(number_of_missing_files > 0):
        print("*** WARNING *** missing {} files.".format(number_of_missing_files))

    new_annotation_data = {
        'info': test_ann['info'],
        'licenses': test_ann['licenses'],
        'images': new_image_sections,
        'annotations': new_annotation_sections,
        'custom_lookup_table': {i: {'image': img, 'caption': txt} for i, (img, txt) in
                                enumerate(zip(target_images, target_captions))}
    }

    with open(os.path.join(root, IMAGE_INDEX_FILE), 'w') as fp:
        json.dump(new_annotation_data, fp)

    # Cleanup temp files
    call("rm {}".format("multilingual_coco_temp_images.txt"), shell=True)
    call("rm {}".format("multilingual_coco_temp_en_captions.txt"), shell=True)


def create_translation_file(index_data, root, lang_code):
    print("Downloading ms_coco translation file", lang_code)
    download_file_name = CAPTIONS_FILE_DOWNLOAD_NAME.format(lang_code)
    download_path = os.path.join(GITHUB_MAIN_PATH, download_file_name)
    captions = _get_downloadable_file("multilingual_coco_temp_en_captions.txt", download_path, False)
    print(captions)

    local_mapping = index_data['custom_lookup_table']
    translation_mapping = {local_mapping[str(i)]['caption']: txt for i, txt in enumerate(captions)}
    with open(os.path.join(root, CAPTIONS_FILE_NAME.format(lang_code)), 'w') as fp:
        json.dump(translation_mapping, fp)

    # Cleanup temp files
    call("rm {}".format("multilingual_coco_temp.txt"), shell=True)
