import os
import sys
from subprocess import call
from collections import defaultdict
import torch
from torchvision.datasets import (
    VisionDataset, ImageFolder,
    CIFAR10, CIFAR100, ImageNet, CocoCaptions, Flickr8k, Flickr30k, Food101, SUN397,
    StanfordCars, FGVCAircraft, DTD, OxfordIIITPet, Caltech101, Flowers102,
    MNIST, STL10, EuroSAT, GTSRB, Kitti, Country211, PCAM, RenderedSST2
)
from . import voc2007, flickr, caltech101, imagenetv2, objectnet
from torch.utils.data import default_collate
from PIL import Image

def build_dataset(dataset_name, root="root", transform=None, split="test", download=True, annotation_file=None, **kwargs):
    """
    Main function to use in order to build a dataset instance,

    dataset_name: str
        name of the dataset
    
    root: str
        root folder where the dataset is downloaded and stored. can be shared among datasets.

    transform: torchvision transform applied to images

    split: str
        split to use, depending on the dataset can have different options.
        In general, `train` and `test` are available.
        For specific splits, please look at the corresponding dataset.
    
    annotation_file: str or None
        only for datasets with captions (used for retrieval) such as COCO
        and Flickr.
    """
    train = (split == "train")
    if dataset_name == "cifar10":
        return CIFAR10(root=root, train=train, transform=transform, download=download, **kwargs)
    elif dataset_name == "cifar100":
        return CIFAR100(root=root, train=train, transform=transform, download=download, **kwargs)
    elif dataset_name == "imagenet1k":
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
            call(f"wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz --output-document={root}/ILSVRC2012_devkit_t12.tar.gz", shell=True)            
            call(f"wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --output-document={root}/ILSVRC2012_img_val.tar", shell=True)            

        ds =  ImageNet(root=root, split="train" if train else "val", transform=transform, **kwargs)
        # use classnames from OpenAI
        ds.classes = classnames["imagenet1k"]
        return ds
    elif dataset_name == "imagenetv2":
        os.makedirs(root, exist_ok=True)
        ds = imagenetv2.ImageNetV2Dataset(variant="matched-frequency", transform=transform, location=root)
        ds.classes = classnames["imagenet1k"]
        return ds   
    elif dataset_name == "imagenet_sketch":
        # Downloadable from https://drive.google.com/open?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA
        if not os.path.exists(root):
            # Automatic download
            print("Downloading imagenet_sketch...")
            if not has_gdown():
                print("GDown is needed to download the dataset. Please install it via `pip install gdown`")
                sys.exit(1)
            # Download ImageNet-Sketch.zip
            call("gdown --id 1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA", shell=True)
            assert os.path.exists("ImageNet-Sketch.zip")
            # Unzip and move to `root`
            call("unzip ImageNet-Sketch.zip", shell=True)
            call(f"mv sketch {root}", shell=True)
        ds = ImageFolder(root=root, transform=transform, **kwargs)
        ds.classes = classnames["imagenet1k"]
        return ds
    elif dataset_name == "imagenet-a":
        # Downloadable from https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
        if not os.path.exists(root):
            print("Downloading imagenet-a...")
            call("wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar", shell=True)
            # Untar and move to `root`
            call("tar xvf imagenet-a.tar", shell=True)
            call(f"mv imagenet-a {root}", shell=True)
        ds = ImageFolder(root=root, transform=transform, **kwargs)
        ds.classes = classnames["imagenet1k"]
        imagenet_a_wnids = ['n01498041', 'n01531178', 'n01534433', 'n01558993', 'n01580077', 'n01614925', 'n01616318', 'n01631663', 'n01641577', 'n01669191', 'n01677366', 'n01687978', 'n01694178', 'n01698640', 'n01735189', 'n01770081', 'n01770393', 'n01774750', 'n01784675', 'n01819313', 'n01820546', 'n01833805', 'n01843383', 'n01847000', 'n01855672', 'n01882714', 'n01910747', 'n01914609', 'n01924916', 'n01944390', 'n01985128', 'n01986214', 'n02007558', 'n02009912', 'n02037110', 'n02051845', 'n02077923', 'n02085620', 'n02099601', 'n02106550', 'n02106662', 'n02110958', 'n02119022', 'n02123394', 'n02127052', 'n02129165', 'n02133161', 'n02137549', 'n02165456', 'n02174001', 'n02177972', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02231487', 'n02233338', 'n02236044', 'n02259212', 'n02268443', 'n02279972', 'n02280649', 'n02281787', 'n02317335', 'n02325366', 'n02346627', 'n02356798', 'n02361337', 'n02410509', 'n02445715', 'n02454379', 'n02486410', 'n02492035', 'n02504458', 'n02655020', 'n02669723', 'n02672831', 'n02676566', 'n02690373', 'n02701002', 'n02730930', 'n02777292', 'n02782093', 'n02787622', 'n02793495', 'n02797295', 'n02802426', 'n02814860', 'n02815834', 'n02837789', 'n02879718', 'n02883205', 'n02895154', 'n02906734', 'n02948072', 'n02951358', 'n02980441', 'n02992211', 'n02999410', 'n03014705', 'n03026506', 'n03124043', 'n03125729', 'n03187595', 'n03196217', 'n03223299', 'n03250847', 'n03255030', 'n03291819', 'n03325584', 'n03355925', 'n03384352', 'n03388043', 'n03417042', 'n03443371', 'n03444034', 'n03445924', 'n03452741', 'n03483316', 'n03584829', 'n03590841', 'n03594945', 'n03617480', 'n03666591', 'n03670208', 'n03717622', 'n03720891', 'n03721384', 'n03724870', 'n03775071', 'n03788195', 'n03804744', 'n03837869', 'n03840681', 'n03854065', 'n03888257', 'n03891332', 'n03935335', 'n03982430', 'n04019541', 'n04033901', 'n04039381', 'n04067472', 'n04086273', 'n04099969', 'n04118538', 'n04131690', 'n04133789', 'n04141076', 'n04146614', 'n04147183', 'n04179913', 'n04208210', 'n04235860', 'n04252077', 'n04252225', 'n04254120', 'n04270147', 'n04275548', 'n04310018', 'n04317175', 'n04344873', 'n04347754', 'n04355338', 'n04366367', 'n04376876', 'n04389033', 'n04399382', 'n04442312', 'n04456115', 'n04482393', 'n04507155', 'n04509417', 'n04532670', 'n04540053', 'n04554684', 'n04562935', 'n04591713', 'n04606251', 'n07583066', 'n07695742', 'n07697313', 'n07697537', 'n07714990', 'n07718472', 'n07720875', 'n07734744', 'n07749582', 'n07753592', 'n07760859', 'n07768694', 'n07831146', 'n09229709', 'n09246464', 'n09472597', 'n09835506', 'n11879895', 'n12057211', 'n12144580', 'n12267677']
        imagenet_a_mask = [wnid in set(imagenet_a_wnids) for wnid in all_imagenet_wordnet_ids]
        ds.classes = [cl for cl, mask in zip(ds.classes, imagenet_a_mask) if mask]
        return ds
    elif dataset_name == "imagenet-r":
        # downloadable from https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
        if not os.path.exists(root):
            print("Downloading imagenet-r...")
            call("wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar", shell=True)
            # Untar and move to `root`
            call("tar xvf imagenet-r.tar", shell=True)
            call(f"mv imagenet-r {root}", shell=True)
        imagenet_r_wnids = {'n01443537', 'n01484850', 'n01494475', 'n01498041', 'n01514859', 'n01518878', 'n01531178', 'n01534433', 'n01614925', 'n01616318', 'n01630670', 'n01632777', 'n01644373', 'n01677366', 'n01694178', 'n01748264', 'n01770393', 'n01774750', 'n01784675', 'n01806143', 'n01820546', 'n01833805', 'n01843383', 'n01847000', 'n01855672', 'n01860187', 'n01882714', 'n01910747', 'n01944390', 'n01983481', 'n01986214', 'n02007558', 'n02009912', 'n02051845', 'n02056570', 'n02066245', 'n02071294', 'n02077923', 'n02085620', 'n02086240', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02091032', 'n02091134', 'n02092339', 'n02094433', 'n02096585', 'n02097298', 'n02098286', 'n02099601', 'n02099712', 'n02102318', 'n02106030', 'n02106166', 'n02106550', 'n02106662', 'n02108089', 'n02108915', 'n02109525', 'n02110185', 'n02110341', 'n02110958', 'n02112018', 'n02112137', 'n02113023', 'n02113624', 'n02113799', 'n02114367', 'n02117135', 'n02119022', 'n02123045', 'n02128385', 'n02128757', 'n02129165', 'n02129604', 'n02130308', 'n02134084', 'n02138441', 'n02165456', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02233338', 'n02236044', 'n02268443', 'n02279972', 'n02317335', 'n02325366', 'n02346627', 'n02356798', 'n02363005', 'n02364673', 'n02391049', 'n02395406', 'n02398521', 'n02410509', 'n02423022', 'n02437616', 'n02445715', 'n02447366', 'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02486410', 'n02510455', 'n02526121', 'n02607072', 'n02655020', 'n02672831', 'n02701002', 'n02749479', 'n02769748', 'n02793495', 'n02797295', 'n02802426', 'n02808440', 'n02814860', 'n02823750', 'n02841315', 'n02843684', 'n02883205', 'n02906734', 'n02909870', 'n02939185', 'n02948072', 'n02950826', 'n02951358', 'n02966193', 'n02980441', 'n02992529', 'n03124170', 'n03272010', 'n03345487', 'n03372029', 'n03424325', 'n03452741', 'n03467068', 'n03481172', 'n03494278', 'n03495258', 'n03498962', 'n03594945', 'n03602883', 'n03630383', 'n03649909', 'n03676483', 'n03710193', 'n03773504', 'n03775071', 'n03888257', 'n03930630', 'n03947888', 'n04086273', 'n04118538', 'n04133789', 'n04141076', 'n04146614', 'n04147183', 'n04192698', 'n04254680', 'n04266014', 'n04275548', 'n04310018', 'n04325704', 'n04347754', 'n04389033', 'n04409515', 'n04465501', 'n04487394', 'n04522168', 'n04536866', 'n04552348', 'n04591713', 'n07614500', 'n07693725', 'n07695742', 'n07697313', 'n07697537', 'n07714571', 'n07714990', 'n07718472', 'n07720875', 'n07734744', 'n07742313', 'n07745940', 'n07749582', 'n07753275', 'n07753592', 'n07768694', 'n07873807', 'n07880968', 'n07920052', 'n09472597', 'n09835506', 'n10565667', 'n12267677'}
        imagenet_r_mask = [wnid in imagenet_r_wnids for wnid in all_imagenet_wordnet_ids]
        ds = ImageFolder(root=root, transform=transform, **kwargs)
        ds.classes = classnames["imagenet1k"]
        ds.classes = [cl for cl, mask in zip(ds.classes, imagenet_r_mask) if mask]
        return ds
    elif dataset_name == "imagenet-o":
        # downloadable from https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar
        if not os.path.exists(root):
            print("Downloading imagenet-o...")
            call("wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar", shell=True)
            # Untar and move to `root`
            call("tar xvf imagenet-o.tar", shell=True)
            call(f"mv imagenet-o {root}", shell=True)
        ds = ImageFolder(root=root, transform=transform, **kwargs)
        ds.classes = classnames["imagenet1k"]
        imagenet_o_wnids = ['n01443537', 'n01704323', 'n01770081', 'n01784675', 'n01819313', 'n01820546', 'n01910747', 'n01917289', 'n01968897', 'n02074367', 'n02317335', 'n02319095', 'n02395406', 'n02454379', 'n02606052', 'n02655020', 'n02666196', 'n02672831', 'n02730930', 'n02777292', 'n02783161', 'n02786058', 'n02787622', 'n02791270', 'n02808304', 'n02817516', 'n02841315', 'n02865351', 'n02877765', 'n02892767', 'n02906734', 'n02910353', 'n02916936', 'n02948072', 'n02965783', 'n03000134', 'n03000684', 'n03017168', 'n03026506', 'n03032252', 'n03075370', 'n03109150', 'n03126707', 'n03134739', 'n03160309', 'n03196217', 'n03207743', 'n03218198', 'n03223299', 'n03240683', 'n03271574', 'n03291819', 'n03297495', 'n03314780', 'n03325584', 'n03344393', 'n03347037', 'n03372029', 'n03376595', 'n03388043', 'n03388183', 'n03400231', 'n03445777', 'n03457902', 'n03467068', 'n03482405', 'n03483316', 'n03494278', 'n03530642', 'n03544143', 'n03584829', 'n03590841', 'n03598930', 'n03602883', 'n03649909', 'n03661043', 'n03666591', 'n03676483', 'n03692522', 'n03706229', 'n03717622', 'n03720891', 'n03721384', 'n03724870', 'n03729826', 'n03733131', 'n03733281', 'n03742115', 'n03786901', 'n03788365', 'n03794056', 'n03804744', 'n03814639', 'n03814906', 'n03825788', 'n03840681', 'n03843555', 'n03854065', 'n03857828', 'n03868863', 'n03874293', 'n03884397', 'n03891251', 'n03908714', 'n03920288', 'n03929660', 'n03930313', 'n03937543', 'n03942813', 'n03944341', 'n03961711', 'n03970156', 'n03982430', 'n03991062', 'n03995372', 'n03998194', 'n04005630', 'n04023962', 'n04033901', 'n04040759', 'n04067472', 'n04074963', 'n04116512', 'n04118776', 'n04125021', 'n04127249', 'n04131690', 'n04141975', 'n04153751', 'n04154565', 'n04201297', 'n04204347', 'n04209133', 'n04209239', 'n04228054', 'n04235860', 'n04243546', 'n04252077', 'n04254120', 'n04258138', 'n04265275', 'n04270147', 'n04275548', 'n04330267', 'n04332243', 'n04336792', 'n04347754', 'n04371430', 'n04371774', 'n04372370', 'n04376876', 'n04409515', 'n04417672', 'n04418357', 'n04423845', 'n04429376', 'n04435653', 'n04442312', 'n04482393', 'n04501370', 'n04507155', 'n04525305', 'n04542943', 'n04554684', 'n04557648', 'n04562935', 'n04579432', 'n04591157', 'n04597913', 'n04599235', 'n06785654', 'n06874185', 'n07615774', 'n07693725', 'n07695742', 'n07697537', 'n07711569', 'n07714990', 'n07715103', 'n07716358', 'n07717410', 'n07718472', 'n07720875', 'n07742313', 'n07745940', 'n07747607', 'n07749582', 'n07753275', 'n07753592', 'n07754684', 'n07768694', 'n07836838', 'n07871810', 'n07873807', 'n07880968', 'n09229709', 'n09472597', 'n12144580', 'n12267677', 'n13052670']
        imagenet_o_mask = [wnid in set(imagenet_o_wnids) for wnid in all_imagenet_wordnet_ids]
        ds.classes = [cl for cl, mask in zip(ds.classes, imagenet_o_mask) if mask]
        return ds
    elif dataset_name == "objectnet":
        # downloadable from https://objectnet.dev/downloads/objectnet-1.0.zip or https://www.dropbox.com/s/raw/cxeztdtm16nzvuw/objectnet-1.0.zip
        if not os.path.exists(root):
            print("Downloading objectnet...")
            call("wget https://objectnet.dev/downloads/objectnet-1.0.zip", shell=True)
            # Untar and move to `root`
            call("UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -P objectnetisatestset objectnet-1.0.zip", shell=True)
            os.makedirs(root)
            call(f"mv objectnet-1.0 {root}", shell=True)
            call(f"cp {root}/objectnet-1.0/mappings/* {root}", shell=True)
        ds = objectnet.ObjectNetDataset(root=root, transform=transform)
        return ds
    elif dataset_name == "voc2007":
        return voc2007.PASCALVoc2007Cropped(root=root, set="train" if train else "test", transform=transform, download=download, **kwargs)
    elif dataset_name == "voc2007_multilabel":
        return voc2007.PASCALVoc2007(root=root, set="train" if train else "test", transform=transform, download=download, **kwargs)
    elif dataset_name == "mscoco_captions":
        # https://gist.github.com/mehdidc/0745a72acb12d3fc9bf91bda65e1ebb6 (annotations)
        # http://images.cocodataset.org/zips/val2014.zip
        if not os.path.exists(root):
            print("Downloading mscoco_captions...")
            call("wget http://images.cocodataset.org/zips/val2014.zip", shell=True)
            call("unzip val2014.zip", shell=True)
            call(f"mv val2014 {root}", shell=True)
        if not os.path.exists(annotation_file):
            # Download COCO Karpathy 5K test set
            annotation_file = f"{root}/coco_test_karpathy.json"
            call(f"wget https://gist.githubusercontent.com/mehdidc/0745a72acb12d3fc9bf91bda65e1ebb6/raw/4e1ab923dea5513280e8c55f7630ca5c0ecbb80a/coco_test_karpathy.json --output-document={annotation_file}", shell=True)
        return CocoCaptions(root=root, annFile=annotation_file, transform=transform, **kwargs)
    elif dataset_name == "flickr30k":
        # downloadable from https://www.kaggle.com/datasets/adityajn105/flickr30k
        # https://gist.github.com/mehdidc/0745a72acb12d3fc9bf91bda65e1ebb6 (annotations)
        # `kaggle datasets download -d adityajn105/flickr30k`
        if not os.path.exists(root):
            # Automatic download
            print("Downloading flickr30k...")
            if not has_kaggle():
                print("Kaggle is needed to download the dataset. Please install it via `pip install kaggle`")
                sys.exit(1)
            call("kaggle datasets download -d adityajn105/flickr30k", shell=True)
            call(f"unzip flickr30k.zip", shell=True)
            call(f"mv Images {root}", shell=True)
            call(f"mv captions.txt {root}", shell=True)
        if not os.path.exists(annotation_file):
            # Download Flickr30K Karpathy test set
            annotation_file = f"{root}/flickr30k_test_karpathy.txt"
            call(f"wget https://gist.githubusercontent.com/mehdidc/0745a72acb12d3fc9bf91bda65e1ebb6/raw/4e1ab923dea5513280e8c55f7630ca5c0ecbb80a/flickr30k_test_karpathy.txt --output-document={annotation_file}", shell=True)
        return flickr.Flickr(root=root, ann_file=annotation_file, transform=transform, **kwargs)
    elif dataset_name == "flickr8k":
        # downloadable from https://www.kaggle.com/datasets/adityajn105/flickr8k
        # `kaggle datasets download -d adityajn105/flickr8k`
        if not os.path.exists(root):
            # Automatic download
            print("Downloading flickr8k...")
            if not has_kaggle():
                print("Kaggle is needed to download the dataset. Please install it via `pip install kaggle`")
                sys.exit(1)
            call("kaggle datasets download -d adityajn105/flickr8k", shell=True)
            call(f"unzip flickr8k.zip", shell=True)
            call(f"mv Images {root}", shell=True)
            call(f"mv captions.txt {root}", shell=True)
        if not os.path.exists(annotation_file):
            # Download Flickr8K Karpathy test set
            annotation_file = f"{root}/flickr8k_test_karpathy.txt"
            call(f"wget https://gist.githubusercontent.com/mehdidc/0745a72acb12d3fc9bf91bda65e1ebb6/raw/6d1d31f8da09310f775905e9ea89aa42d0739f22/flickr8k_test_karpathy.txt --output-document={annotation_file}", shell=True)
        return flickr.Flickr(root=root, ann_file=annotation_file, transform=transform, **kwargs)
    elif dataset_name == "food101":
        ds = Food101(root=root, split="train" if train else "test", transform=transform, download=download, **kwargs)
        # we use the default class names, we just  replace "_" by spaces
        # to delimit words
        ds.classes = [cl.replace("_", " ") for cl in ds.classes]
        return ds
    elif dataset_name == "sun397":
        # we use the default class names, we just  replace "_" and "/" by spaces
        # to delimit words
        ds = SUN397(root=root, transform=transform, download=download, **kwargs)
        ds.classes = [cl.replace("_", " ").replace("/", " ") for cl in ds.classes]
        return ds
    elif dataset_name == "cars":
        return StanfordCars(root=root, split="train" if train else "test", transform=transform, download=download, **kwargs)
    elif dataset_name == "fgvc_aircraft":
        return FGVCAircraft(root=root, annotation_level="variant", split="train" if train else "test", transform=transform, download=download, **kwargs)
    elif dataset_name == "dtd":
        return DTD(root=root, split="train" if train else "test", transform=transform, download=download, **kwargs)
    elif dataset_name == "pets":
        return OxfordIIITPet(root=root, split="train" if train else "test", target_types="category", transform=transform, download=download, **kwargs)
    elif dataset_name == "caltech101":
        # broken download link (can't download google drive), fixed by this PR https://github.com/pytorch/vision/pull/5645
        # also available in "vtab/caltech101" using VTAB splits, we advice to use VTAB version rather than this one 
        # since in this one (torchvision) there are no pre-defined test splits
        ds = caltech101.Caltech101(root=root, target_type="category", transform=transform, download=download, **kwargs)
        ds.classes = classnames["caltech101"]
        return ds
    elif dataset_name == "flowers":
        ds = Flowers102(root=root, split="train" if train else "test", transform=transform, download=download, **kwargs)
        # class indices started by 1 until it was fixed in  a  PR (#TODO link of the PR)
        # if older torchvision version, fix it using a target transform that decrements label index 
        # TODO figure out minimal torchvision version needed instead of decrementing
        if ds[0][1] == 1:
            ds.target_transform = lambda y:y-1
        ds.classes = classnames["flowers"]
        return ds
    elif dataset_name == "mnist":
        ds = MNIST(root=root, train=train, transform=transform, download=download, **kwargs)
        ds.classes = classnames["mnist"]
        return ds
    elif dataset_name == "stl10":
        return STL10(root=root, split="train" if train else "test", transform=transform, download=download, **kwargs)
    elif dataset_name == "eurosat":
        ds = EuroSAT(root=root, transform=transform, download=download, **kwargs)
        ds.classes = classnames["eurosat"]
        return ds
    elif dataset_name == "gtsrb":
        ds =  GTSRB(root=root, split="train" if train else "test", transform=transform, download=download, **kwargs)
        ds.classes = classnames["gtsrb"]
        return ds
    elif dataset_name == "country211":
        ds = Country211(root=root, split="train" if train else "test", transform=transform, download=download, **kwargs)
        ds.classes = classnames["country211"]
        return ds
    elif dataset_name == "pcam":
        # Dead link. Fixed by this PR on torchvision https://github.com/pytorch/vision/pull/5645
        # TODO figure out minimal torchvision version needed
        ds =  PCAM(root=root, split="train" if train else "test", transform=transform, download=download, **kwargs)
        ds.classes = classnames["pcam"]
        return ds
    elif dataset_name == "renderedsst2":
        return RenderedSST2(root=root, split="train" if train else "test", transform=transform, download=download, **kwargs)
    elif dataset_name == "fer2013":
        # Downloadable from  https://www.kaggle.com/datasets/msambare/fer2013
        # `kaggle datasets download -d msambare/fer2013`
        if not os.path.exists(root):
            # Automatic download
            print("Downloading fer2013...")
            if not has_kaggle():
                print("Kaggle is needed to download the dataset. Please install it via `pip install kaggle`")
                sys.exit(1)
            call("kaggle datasets download -d msambare/fer2013", shell=True)
            call(f"unzip fer2013.zip -d {root}", shell=True)
        root = os.path.join(root, "train" if train else "test")
        ds = ImageFolder(root=root, transform=transform)
        ds.classes = classnames["fer2013"]
        return ds
    elif dataset_name.startswith("tfds/"):
        # TFDS datasets support using `timm` and `tensorflow_datasets`
        prefix, *name_list = dataset_name.split("/")
        name = "/".join(name_list)
        return build_tfds_dataset(name, download=download, split=split, data_dir=root, transform=transform)
    elif dataset_name.startswith("vtab/"):
        # VTAB datasets support using `tensorflow_datasets` and `task_adaptation`
        prefix, *name_list = dataset_name.split("/")
        name = "/".join(name_list)
        return build_vtab_dataset(name, download=download, split=split, data_dir=root, transform=transform)
    elif dataset_name == "dummy":
        return Dummy()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}.")

class Dummy():

    def __init__(self):
        self.classes = ["blank image", "noisy image"]

    def __getitem__(self, i):
        return torch.zeros(3,224,224), 0

    def __len__(self):
        return 1

def get_dataset_collate_fn(dataset_name):
    if dataset_name in ("mscoco_captions", "flickr30k", "flickr8k"):
        return image_captions_collate_fn
    else:
        return default_collate

def has_gdown():
    return call("which gdown", shell=True) == 0

def has_kaggle():
    return call("which kaggle", shell=True) == 0


def build_vtab_dataset(dataset_name, transform, download=True, split="test", data_dir="root"):
    # Using VTAB splits instead of default TFDS splits
    from .tfds import VTABIterableDataset, disable_gpus_on_tensorflow, download_tfds_dataset

    # avoid Tensorflow owning GPUs to not clash with PyTorch
    disable_gpus_on_tensorflow()

    # by default we take classes from TFDS (default behavior if `classes` stays None),
    # except for the datasets that will override `classes` (e.g., clevr_*)
    classes = None
    if dataset_name == "caltech101":
        from task_adaptation.data.caltech import Caltech101
        tfds_dataset = Caltech101(data_dir=data_dir)
        classes = classnames["caltech101_vtab"]
    elif dataset_name == "cars":
        from task_adaptation.data.cars import CarsData
        tfds_dataset = CarsData(data_dir=data_dir)
    elif dataset_name in ("cifar10", "cifar100"):
        from task_adaptation.data.cifar import CifarData
        tfds_dataset = CifarData(data_dir=data_dir, num_classes=10 if dataset_name == "cifar10" else 100)
    elif dataset_name.startswith("clevr_"):
        from task_adaptation.data.clevr import CLEVRData
        task = _extract_task(dataset_name)
        assert task in ("count_all", "closest_object_distance")
        tfds_dataset = CLEVRData(task=task, data_dir=data_dir)
        if task == "count_all":
            classes = classnames["clevr_count_all"]
        elif task == "closest_object_distance":
            classes = classnames["clevr_closest_object_distance"]
        else:
            raise ValueError(f"non supported: {task}")
    elif dataset_name == "cub":
        from task_adaptation.data.cub import CUB2011Data
        tfds_dataset = CUB2011Data(data_dir=data_dir)
    elif dataset_name == "diabetic_retinopathy":
        # Needs manual download from Kaggle
        # 1) `kaggle competitions download -c diabetic-retinopathy-detection` on $ROOT/downloads/manual
        # 2) extract archives  on $ROOT/downloads/manual
        if not os.path.exists(data_dir):
            # Automatic download
            print("Downloading diabetic_retinopathy...")
            if not has_kaggle():
                print("Kaggle is needed to download the dataset. Please install it via `pip install kaggle`")
                sys.exit(1)
            os.makedirs(os.path.join(data_dir, "downloads", "manual"))
            call(f"kaggle competitions download -c diabetic-retinopathy-detection -p {data_dir}/downloads/manual", shell=True)
            call(f"cd {data_dir}/downloads/manual;unzip diabetic-retinopathy-detection.zip;cat train.zip*>train.zip;cat test.zip*>test.zip;unzip train.zip; unzip test.zip;unzip sample.zip;unzip trainLabels.csv.zip", shell=True)
        from task_adaptation.data.diabetic_retinopathy import RetinopathyData
        tfds_dataset = RetinopathyData(config="btgraham-300", data_dir=data_dir)
        classes = classnames["diabetic_retinopathy"]
    elif dataset_name == "dmlab":
        from task_adaptation.data.dmlab import DmlabData
        download_tfds_dataset("dmlab", data_dir=data_dir) # it's not called in the original VTAB code, so we do it explictly
        tfds_dataset = DmlabData(data_dir=data_dir)
        classes = classnames["dmlab"]
    elif dataset_name.startswith("dsprites_"):
        from task_adaptation.data.dsprites import DSpritesData
        task = _extract_task(dataset_name)
        assert task in ("label_shape", "label_scale", "label_orientation", "label_x_position", "label_y_position")
        tfds_dataset = DSpritesData(task, data_dir=data_dir)
        classes = tfds_dataset._dataset_builder.info.features[task].names
    elif dataset_name == "dtd":
        from task_adaptation.data.dtd import DTDData
        tfds_dataset = DTDData(data_dir=data_dir)
    elif dataset_name == "eurosat":
        from task_adaptation.data.eurosat import EurosatData
        tfds_dataset = EurosatData(subset="rgb", data_key="image", data_dir=data_dir)
        classes = classnames["eurosat"]
    elif dataset_name == "food101":
        from task_adaptation.data.food101 import Food101Data
        tfds_dataset = Food101Data(data_dir=data_dir)
    elif dataset_name == "inaturalist":
        from task_adaptation.data.inaturalist import INaturalistData
        tfds_dataset = INaturalistData(data_dir=data_dir, year=2017)
    elif dataset_name.startswith("kitti_"):
        from .kitti import KittiData
        task = _extract_task(dataset_name)
        assert task in (
            "count_all", "count_left", "count_far", "count_near", 
            "closest_object_distance", "closest_object_x_location", 
            "count_vehicles", "closest_vehicle_distance",
        )
        tfds_dataset = KittiData(task=task, data_dir=data_dir)
        if task == "closest_vehicle_distance":
            classes = classnames["kitti_closest_vehicle_distance"]
        else:
            raise ValueError(f"Unsupported task: {task}")
    elif dataset_name == "flowers":
        from task_adaptation.data.oxford_flowers102 import OxfordFlowers102Data
        tfds_dataset = OxfordFlowers102Data(data_dir=data_dir)
    elif dataset_name == "pets":
        from task_adaptation.data.oxford_iiit_pet import OxfordIIITPetData
        tfds_dataset = OxfordIIITPetData(data_dir=data_dir)
        classes = classnames["pets"]
    elif dataset_name == "pcam":
        from task_adaptation.data.patch_camelyon import PatchCamelyonData
        tfds_dataset = PatchCamelyonData(data_dir=data_dir)
        classes = classnames["pcam"]
    elif dataset_name == "resisc45":
        # Needs download from OneDrive: https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs
        # The archive needs to to be put at <DATASET_ROOT>/downloads/manual then extracted
        if not os.path.exists(data_dir):
            os.makedirs(os.path.join(data_dir, "downloads", "manual"))
            call(f"wget 'https://onedrive.live.com/download?resid=5C5E061130630A68!107&authkey=!AHHNaHIlzp_IXjs' --output-document={data_dir}/downloads/manual/resisc45.rar", shell=True)
            call(f"cd {data_dir}/downloads/manual;unrar x resisc45.rar", shell=True)
        from task_adaptation.data.resisc45 import Resisc45Data
        tfds_dataset = Resisc45Data(data_dir=data_dir)
    elif dataset_name.startswith("smallnorb_"):
        from task_adaptation.data.smallnorb import SmallNORBData
        task = _extract_task(dataset_name)
        assert task in ("label_category", "label_elevation", "label_azimuth", "label_lighting")
        tfds_dataset = SmallNORBData(predicted_attribute=task, data_dir=data_dir)
        classes = tfds_dataset._dataset_builder.info.features[task].names
    elif dataset_name == "sun397":
        from task_adaptation.data.sun397 import Sun397Data
        #FIXME There is a problem in `sun397`, when TFDS tries download it
        # there is an image that cannot be decoded. For the time being
        # we will use torchvision's SUN397 instead.
        tfds_dataset = Sun397Data(config="tfds", data_dir=data_dir)
    elif dataset_name == "svhn":
        from task_adaptation.data.svhn import SvhnData
        tfds_dataset = SvhnData(data_dir=data_dir)
        classes = classnames["svhn"]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    ds =  VTABIterableDataset(
        tfds_dataset, 
        input_name="image", label_name="label", 
        transform=transform, 
        target_transform=int,
        split=split,
        classes=classes,
    )
    return ds

def build_tfds_dataset(name, transform, download=True, split="test", data_dir="root", classes=None):
    from .tfds import disable_gpus_on_tensorflow
    disable_gpus_on_tensorflow()
    import tensorflow_datasets as tfds
    import timm
    builder = tfds.builder(name, data_dir=data_dir)
    if download:
        builder.download_and_prepare()
    splits = list(builder.info.splits.keys())
    assert split in splits, (split, splits)
    ds = timm.data.create_dataset(f"tfds/{name}", data_dir, split=split, transform=transform, target_transform=int)
    ds.classes = builder.info.features['label'].names if classes is None else classes
    return ds


def _extract_task(dataset_name):
    prefix, *task_name_list = dataset_name.split("_")
    task = "_".join(task_name_list)
    return task


def image_captions_collate_fn(batch):
    transposed = list(zip(*batch))
    imgs = default_collate(transposed[0])
    texts = transposed[1]
    return imgs, texts


def get_zeroshot_classification_templates(dataset_name):
    if dataset_name.startswith("tfds/") or dataset_name.startswith("vtab/"):
        name = dataset_name.split("/")[1]
    else:
        name = dataset_name
    return zeroshot_classification_templates.get(name, DEFAULT_ZEROSHOT_CLASSIFICATION_TEMPLATES)

# Zero-shot classification templates, collected from a bunch of sources
# - CLIP paper (https://github.com/openai/CLIP/blob/main/data/prompts.md)
# - Lit Paper (https://arxiv.org/pdf/2111.07991.pdf)
# - SLIP paper (https://github.com/facebookresearch/SLIP/blob/main/templates.json)
# Some are fixed mnaually

zeroshot_classification_templates = {
    "cifar10": [
        "a photo of a {c}.",
        "a blurry photo of a {c}.",
        "a black and white photo of a {c}.",
        "a low contrast photo of a {c}.",
        "a high contrast photo of a {c}.",
        "a bad photo of a {c}.",
        "a good photo of a {c}.",
        "a photo of a small {c}.",
        "a photo of a big {c}.",
        "a photo of the {c}.",
        "a blurry photo of the {c}.",
        "a black and white photo of the {c}.",
        "a low contrast photo of the {c}.",
        "a high contrast photo of the {c}.",
        "a bad photo of the {c}.",
        "a good photo of the {c}.",
        "a photo of the small {c}.",
        "a photo of the big {c}."
    ],
    "cifar100":[
        "a photo of a {c}.",
        "a blurry photo of a {c}.",
        "a black and white photo of a {c}.",
        "a low contrast photo of a {c}.",
        "a high contrast photo of a {c}.",
        "a bad photo of a {c}.",
        "a good photo of a {c}.",
        "a photo of a small {c}.",
        "a photo of a big {c}.",
        "a photo of the {c}.",
        "a blurry photo of the {c}.",
        "a black and white photo of the {c}.",
        "a low contrast photo of the {c}.",
        "a high contrast photo of the {c}.",
        "a bad photo of the {c}.",
        "a good photo of the {c}.",
        "a photo of the small {c}.",
        "a photo of the big {c}."
    ],
    "imagenet1k": [
        "a bad photo of a {c}.",
        "a photo of many {c}.",
        "a sculpture of a {c}.",
        "a photo of the hard to see {c}.",
        "a low resolution photo of the {c}.",
        "a rendering of a {c}.",
        "graffiti of a {c}.",
        "a bad photo of the {c}.",
        "a cropped photo of the {c}.",
        "a tattoo of a {c}.",
        "the embroidered {c}.",
        "a photo of a hard to see {c}.",
        "a bright photo of a {c}.",
        "a photo of a clean {c}.",
        "a photo of a dirty {c}.",
        "a dark photo of the {c}.",
        "a drawing of a {c}.",
        "a photo of my {c}.",
        "the plastic {c}.",
        "a photo of the cool {c}.",
        "a close-up photo of a {c}.",
        "a black and white photo of the {c}.",
        "a painting of the {c}.",
        "a painting of a {c}.",
        "a pixelated photo of the {c}.",
        "a sculpture of the {c}.",
        "a bright photo of the {c}.",
        "a cropped photo of a {c}.",
        "a plastic {c}.",
        "a photo of the dirty {c}.",
        "a jpeg corrupted photo of a {c}.",
        "a blurry photo of the {c}.",
        "a photo of the {c}.",
        "a good photo of the {c}.",
        "a rendering of the {c}.",
        "a {c} in a video game.",
        "a photo of one {c}.",
        "a doodle of a {c}.",
        "a close-up photo of the {c}.",
        "a photo of a {c}.",
        "the origami {c}.",
        "the {c} in a video game.",
        "a sketch of a {c}.",
        "a doodle of the {c}.",
        "a origami {c}.",
        "a low resolution photo of a {c}.",
        "the toy {c}.",
        "a rendition of the {c}.",
        "a photo of the clean {c}.",
        "a photo of a large {c}.",
        "a rendition of a {c}.",
        "a photo of a nice {c}.",
        "a photo of a weird {c}.",
        "a blurry photo of a {c}.",
        "a cartoon {c}.",
        "art of a {c}.",
        "a sketch of the {c}.",
        "a embroidered {c}.",
        "a pixelated photo of a {c}.",
        "itap of the {c}.",
        "a jpeg corrupted photo of the {c}.",
        "a good photo of a {c}.",
        "a plushie {c}.",
        "a photo of the nice {c}.",
        "a photo of the small {c}.",
        "a photo of the weird {c}.",
        "the cartoon {c}.",
        "art of the {c}.",
        "a drawing of the {c}.",
        "a photo of the large {c}.",
        "a black and white photo of a {c}.",
        "the plushie {c}.",
        "a dark photo of a {c}.",
        "itap of a {c}.",
        "graffiti of the {c}.",
        "a toy {c}.",
        "itap of my {c}.",
        "a photo of a cool {c}.",
        "a photo of a small {c}.",
        "a tattoo of the {c}."
    ],
    "food101":[
        'a photo of {c}, a type of food.'
    ],
    "sun397":[
        'a photo of a {c}.',
        'a photo of the {c}.',
    ],
    "cars":[
        'a photo of a {c}.',
        'a photo of the {c}.',
        'a photo of my {c}.',
        'i love my {c}!',
        'a photo of my dirty {c}.',
        'a photo of my clean {c}.',
        'a photo of my new {c}.',
        'a photo of my old {c}.',
    ],
    "fgvc_aircraft":[
        'a photo of a {c}, a type of aircraft.',
        'a photo of the {c}, a type of aircraft.',
    ],
    "dtd":[
        'a photo of a {c} texture.',
        'a photo of a {c} pattern.',
        'a photo of a {c} thing.',
        'a photo of a {c} object.',
        'a photo of the {c} texture.',
        'a photo of the {c} pattern.',
        'a photo of the {c} thing.',
        'a photo of the {c} object.',    
    ],
    "pets":[
        'a photo of a {c}, a type of pet.',
    ],
    "caltech101":[
        'a photo of a {c}.',
        'a painting of a {c}.',
        'a plastic {c}.',
        'a sculpture of a {c}.',
        'a sketch of a {c}.',
        'a tattoo of a {c}.',
        'a toy {c}.',
        'a rendition of a {c}.',
        'a embroidered {c}.',
        'a cartoon {c}.',
        'a {c} in a video game.',
        'a plushie {c}.',
        'a origami {c}.',
        'art of a {c}.',
        'graffiti of a {c}.',
        'a drawing of a {c}.',
        'a doodle of a {c}.',
        'a photo of the {c}.',
        'a painting of the {c}.',
        'the plastic {c}.',
        'a sculpture of the {c}.',
        'a sketch of the {c}.',
        'a tattoo of the {c}.',
        'the toy {c}.',
        'a rendition of the {c}.',
        'the embroidered {c}.',
        'the cartoon {c}.',
        'the {c} in a video game.',
        'the plushie {c}.',
        'the origami {c}.',
        'art of the {c}.',
        'graffiti of the {c}.',
        'a drawing of the {c}.',
        'a doodle of the {c}.',
    ],
    "flowers":[
        'a photo of a {c}, a type of flower.',
    ],
    "mnist": [
        'a photo of the number: "{c}".',
    ],
    "stl10": [
        'a photo of a {c}.',
        'a photo of the {c}.',
    ],
    "eurosat":[
        'a centered satellite photo of {c}.',
        'a centered satellite photo of a {c}.',
        'a centered satellite photo of the {c}.',
    ],
    "gtsrb":[
        'a zoomed in photo of a "{c}" traffic sign.',
        'a centered photo of a "{c}" traffic sign.',
        'a close up photo of a "{c}" traffic sign.',
    ],
    "country211":[
        'a photo i took in {c}.',
        'a photo i took while visiting {c}.',
        'a photo from my home country of {c}.',
        'a photo from my visit to {c}.',
        'a photo showing the country of {c}.',
    ],
    "renderedsst2":[
        'a {c} review of a movie.',
    ],
    "voc2007":[
        'a photo of a {c}.',
    ],
    "voc2007_multilabel":[
        'a photo of a {c}.',
    ],
    "fer2013":[
        'a photo of a {c} looking face.',
        'a photo of a face showing the emotion: {c}.',
        'a photo of a face looking {c}.',
        'a face that looks {c}.',
        'they look {c}.',
        'look at how {c} they are.',
    ],
    "clevr_count_all":[
        "a picture of {c} objects"
    ],
    "clevr_closest_object_distance":[
        "{c} shapes."
    ],
    "pcam":[
        "a histopathology slide showing {c}",
        "histopathology image of {c}"
    ],
    "svhn":[
        "a photo of the number {c} written on a sign",
        "an outdoor house number {c}",
        "the number {c} in the center of the image",
        "an outdoor number {c} writte on a sign",
        "an outdoor number {c}",
        "a centered image of the number {c}",
    ],
    "resisc45":[
        "a sattelite image of {c}",
        "an aerial view of {c}",
        "a sattelite photo of {c}",
        "{c} from above",
    ],
    "kitti_closest_vehicle_distance":[
        "{c}"
    ],
    "smallnorb_label_azimuth":[
        "an object rotated at {c}",
        "something rotated at {c}",
        "{c} rotation",
        "something at a {c} angle",
    ],
    "smallnorb_label_elevation":[
        "an object rotated at {c}",
        "something rotated at {c}",
        "{c} rotation",
        "something at a {c} angle",
    ],
    "dsprites_label_x_position": [
        "an object located at position {c}% on the horizontal axis",
    ],
    "dsprites_label_orientation":[
        "an object rotated at {c}",
        "something rotated at {c}",
        "{c} rotation",
        "something at a {c} angle",
    ],
    "dmlab":[
        "{c}"
    ],
    "diabetic_retinopathy":[
        "a retinal image with {c}",
    ],
    "dummy":[
        "a photo of a {c}"
    ],
}


# Class names for different datasets
# In general, we use the default class names from torchvision or VTAB/TFDS,
# except for the datasets defined in `classnames`
# These classnames are collected from various sources:
# - CLIP paper (https://github.com/openai/CLIP/blob/main/data/prompts.md)
# - Lit Paper (https://arxiv.org/pdf/2111.07991.pdf)
# - SLIP paper (https://github.com/facebookresearch/SLIP/blob/main/templates.json)
# Some are fixed manually

classnames = dict(
    flowers = [
        'pink primrose',
        'hard-leaved pocket orchid',
        'canterbury bells',
        'sweet pea',
        'english marigold',
        'tiger lily',
        'moon orchid',
        'bird of paradise',
        'monkshood',
        'globe thistle',
        'snapdragon',
        "colt's foot",
        'king protea',
        'spear thistle',
        'yellow iris',
        'globe flower',
        'purple coneflower',
        'peruvian lily',
        'balloon flower',
        'giant white arum lily',
        'fire lily',
        'pincushion flower',
        'fritillary',
        'red ginger',
        'grape hyacinth',
        'corn poppy',
        'prince of wales feathers',
        'stemless gentian',
        'artichoke',
        'sweet william',
        'carnation',
        'garden phlox',
        'love in the mist',
        'mexican aster',
        'alpine sea holly',
        'ruby-lipped cattleya',
        'cape flower',
        'great masterwort',
        'siam tulip',
        'lenten rose',
        'barbeton daisy',
        'daffodil',
        'sword lily',
        'poinsettia',
        'bolero deep blue',
        'wallflower',
        'marigold',
        'buttercup',
        'oxeye daisy',
        'common dandelion',
        'petunia',
        'wild pansy',
        'primula',
        'sunflower',
        'pelargonium',
        'bishop of llandaff',
        'gaura',
        'geranium',
        'orange dahlia',
        'pink and yellow dahlia',
        'cautleya spicata',
        'japanese anemone',
        'black-eyed susan',
        'silverbush',
        'californian poppy',
        'osteospermum',
        'spring crocus',
        'bearded iris',
        'windflower',
        'tree poppy',
        'gazania',
        'azalea',
        'water lily',
        'rose',
        'thorn apple',
        'morning glory',
        'passion flower',
        'lotus',
        'toad lily',
        'anthurium',
        'frangipani',
        'clematis',
        'hibiscus',
        'columbine',
        'desert-rose',
        'tree mallow',
        'magnolia',
        'cyclamen',
        'watercress',
        'canna lily',
        'hippeastrum',
        'bee balm',
        'air plant',
        'foxglove',
        'bougainvillea',
        'camellia',
        'mallow',
        'mexican petunia',
        'bromelia',
        'blanket flower',
        'trumpet creeper',
        'blackberry lily',
    ],
    gtsrb= [
        'red and white circle 20 kph speed limit',
        'red and white circle 30 kph speed limit',
        'red and white circle 50 kph speed limit',
        'red and white circle 60 kph speed limit',
        'red and white circle 70 kph speed limit',
        'red and white circle 80 kph speed limit',
        'end / de-restriction of 80 kph speed limit',
        'red and white circle 100 kph speed limit',
        'red and white circle 120 kph speed limit',
        'red and white circle red car and black car no passing',
        'red and white circle red truck and black car no passing',
        'red and white triangle road intersection warning',
        'white and yellow diamond priority road',
        'red and white upside down triangle yield right-of-way',
        'stop',
        'empty red and white circle',
        'red and white circle no truck entry',
        'red circle with white horizonal stripe no entry',
        'red and white triangle with exclamation mark warning',
        'red and white triangle with black left curve approaching warning',
        'red and white triangle with black right curve approaching warning',
        'red and white triangle with black double curve approaching warning',
        'red and white triangle rough / bumpy road warning',
        'red and white triangle car skidding / slipping warning',
        'red and white triangle with merging / narrow lanes warning',
        'red and white triangle with person digging / construction / road work warning',
        'red and white triangle with traffic light approaching warning',
        'red and white triangle with person walking warning',
        'red and white triangle with child and person walking warning',
        'red and white triangle with bicyle warning',
        'red and white triangle with snowflake / ice warning',
        'red and white triangle with deer warning',
        'white circle with gray strike bar no speed limit',
        'blue circle with white right turn arrow mandatory',
        'blue circle with white left turn arrow mandatory',
        'blue circle with white forward arrow mandatory',
        'blue circle with white forward or right turn arrow mandatory',
        'blue circle with white forward or left turn arrow mandatory',
        'blue circle with white keep right arrow mandatory',
        'blue circle with white keep left arrow mandatory',
        'blue circle with white arrows indicating a traffic circle',
        'white circle with gray strike bar indicating no passing for cars has ended',
        'white circle with gray strike bar indicating no passing for trucks has ended',
    ],
    country211 = [
        'Andorra',
        'United Arab Emirates',
        'Afghanistan',
        'Antigua and Barbuda',
        'Anguilla',
        'Albania',
        'Armenia',
        'Angola',
        'Antarctica',
        'Argentina',
        'Austria',
        'Australia',
        'Aruba',
        'Aland Islands',
        'Azerbaijan',
        'Bosnia and Herzegovina',
        'Barbados',
        'Bangladesh',
        'Belgium',
        'Burkina Faso',
        'Bulgaria',
        'Bahrain',
        'Benin',
        'Bermuda',
        'Brunei Darussalam',
        'Bolivia',
        'Bonaire, Saint Eustatius and Saba',
        'Brazil',
        'Bahamas',
        'Bhutan',
        'Botswana',
        'Belarus',
        'Belize',
        'Canada',
        'DR Congo',
        'Central African Republic',
        'Switzerland',
        "Cote d'Ivoire",
        'Cook Islands',
        'Chile',
        'Cameroon',
        'China',
        'Colombia',
        'Costa Rica',
        'Cuba',
        'Cabo Verde',
        'Curacao',
        'Cyprus',
        'Czech Republic',
        'Germany',
        'Denmark',
        'Dominica',
        'Dominican Republic',
        'Algeria',
        'Ecuador',
        'Estonia',
        'Egypt',
        'Spain',
        'Ethiopia',
        'Finland',
        'Fiji',
        'Falkland Islands',
        'Faeroe Islands',
        'France',
        'Gabon',
        'United Kingdom',
        'Grenada',
        'Georgia',
        'French Guiana',
        'Guernsey',
        'Ghana',
        'Gibraltar',
        'Greenland',
        'Gambia',
        'Guadeloupe',
        'Greece',
        'South Georgia and South Sandwich Is.',
        'Guatemala',
        'Guam',
        'Guyana',
        'Hong Kong',
        'Honduras',
        'Croatia',
        'Haiti',
        'Hungary',
        'Indonesia',
        'Ireland',
        'Israel',
        'Isle of Man',
        'India',
        'Iraq',
        'Iran',
        'Iceland',
        'Italy',
        'Jersey',
        'Jamaica',
        'Jordan',
        'Japan',
        'Kenya',
        'Kyrgyz Republic',
        'Cambodia',
        'St. Kitts and Nevis',
        'North Korea',
        'South Korea',
        'Kuwait',
        'Cayman Islands',
        'Kazakhstan',
        'Laos',
        'Lebanon',
        'St. Lucia',
        'Liechtenstein',
        'Sri Lanka',
        'Liberia',
        'Lithuania',
        'Luxembourg',
        'Latvia',
        'Libya',
        'Morocco',
        'Monaco',
        'Moldova',
        'Montenegro',
        'Saint-Martin',
        'Madagascar',
        'Macedonia',
        'Mali',
        'Myanmar',
        'Mongolia',
        'Macau',
        'Martinique',
        'Mauritania',
        'Malta',
        'Mauritius',
        'Maldives',
        'Malawi',
        'Mexico',
        'Malaysia',
        'Mozambique',
        'Namibia',
        'New Caledonia',
        'Nigeria',
        'Nicaragua',
        'Netherlands',
        'Norway',
        'Nepal',
        'New Zealand',
        'Oman',
        'Panama',
        'Peru',
        'French Polynesia',
        'Papua New Guinea',
        'Philippines',
        'Pakistan',
        'Poland',
        'Puerto Rico',
        'Palestine',
        'Portugal',
        'Palau',
        'Paraguay',
        'Qatar',
        'Reunion',
        'Romania',
        'Serbia',
        'Russia',
        'Rwanda',
        'Saudi Arabia',
        'Solomon Islands',
        'Seychelles',
        'Sudan',
        'Sweden',
        'Singapore',
        'St. Helena',
        'Slovenia',
        'Svalbard and Jan Mayen Islands',
        'Slovakia',
        'Sierra Leone',
        'San Marino',
        'Senegal',
        'Somalia',
        'South Sudan',
        'El Salvador',
        'Sint Maarten',
        'Syria',
        'Eswatini',
        'Togo',
        'Thailand',
        'Tajikistan',
        'Timor-Leste',
        'Turkmenistan',
        'Tunisia',
        'Tonga',
        'Turkey',
        'Trinidad and Tobago',
        'Taiwan',
        'Tanzania',
        'Ukraine',
        'Uganda',
        'United States',
        'Uruguay',
        'Uzbekistan',
        'Vatican',
        'Venezuela',
        'British Virgin Islands',
        'United States Virgin Islands',
        'Vietnam',
        'Vanuatu',
        'Samoa',
        'Kosovo',
        'Yemen',
        'South Africa',
        'Zambia',
        'Zimbabwe',
    ],
    eurosat = [
        'annual crop land',
        'forest',
        'brushland or shrubland',
        'highway or road',
        'industrial buildings or commercial buildings',
        'pasture land',
        'permanent crop land',
        'residential buildings or homes or apartments',
        'river',
        'lake or sea',
    ],
    fer2013 = [
        "angry",
        "disgusted",
        "fearful",
        "happy",
        "neutral",
        "sad",
        "surprised",
    ],
    caltech101 = [
        'background',
        'off-center face',
        'centered face',
        'leopard',
        'motorbike',
        'accordion',
        'airplane',
        'anchor',
        'ant',
        'barrel',
        'bass',
        'beaver',
        'binocular',
        'bonsai',
        'brain',
        'brontosaurus',
        'buddha',
        'butterfly',
        'camera',
        'cannon',
        'side of a car',
        'ceiling fan',
        'cellphone',
        'chair',
        'chandelier',
        'body of a cougar cat',
        'face of a cougar cat',
        'crab',
        'crayfish',
        'crocodile',
        'head of a  crocodile',
        'cup',
        'dalmatian',
        'dollar bill',
        'dolphin',
        'dragonfly',
        'electric guitar',
        'elephant',
        'emu',
        'euphonium',
        'ewer',
        'ferry',
        'flamingo',
        'head of a flamingo',
        'garfield',
        'gerenuk',
        'gramophone',
        'grand piano',
        'hawksbill',
        'headphone',
        'hedgehog',
        'helicopter',
        'ibis',
        'inline skate',
        'joshua tree',
        'kangaroo',
        'ketch',
        'lamp',
        'laptop',
        'llama',
        'lobster',
        'lotus',
        'mandolin',
        'mayfly',
        'menorah',
        'metronome',
        'minaret',
        'nautilus',
        'octopus',
        'okapi',
        'pagoda',
        'panda',
        'pigeon',
        'pizza',
        'platypus',
        'pyramid',
        'revolver',
        'rhino',
        'rooster',
        'saxophone',
        'schooner',
        'scissors',
        'scorpion',
        'sea horse',
        'snoopy (cartoon beagle)',
        'soccer ball',
        'stapler',
        'starfish',
        'stegosaurus',
        'stop sign',
        'strawberry',
        'sunflower',
        'tick',
        'trilobite',
        'umbrella',
        'watch',
        'water lilly',
        'wheelchair',
        'wild cat',
        'windsor chair',
        'wrench',
        'yin and yang symbol',
    ],
    # same as `caltech1101`, just a different ordering`
    caltech101_vtab = [ 
    'accordion', 'airplane', 'anchor', 'ant', 'background', 'barrel', 'bass', 'beaver', 'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera', 'cannon', 'side of a car', 'ceiling fan', 'cellphone', 'chair', 'chandelier', 'body of a cougar cat', 'face of a cougar cat', 'crab', 'crayfish', 'crocodile', 'head of a  crocodile', 'cup', 'dalmatian', 'dollar bill', 'dolphin', 'dragonfly', 'electric guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'off-center face', 'centered face', 'ferry', 'flamingo', 'head of a flamingo', 'garfield', 'gerenuk', 'gramophone', 'grand piano', 'hawksbill', 'headphone', 'hedgehog', 'helicopter', 'ibis', 'inline skate', 'joshua tree', 'kangaroo', 'ketch', 'lamp', 'laptop', 'leopard', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome', 'minaret', 'motorbike', 'nautilus', 'octopus', 'okapi', 'pagoda', 'panda', 'pigeon', 'pizza', 'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 'schooner', 'scissors', 'scorpion', 'sea horse', 'snoopy (cartoon beagle)', 'soccer ball', 'stapler', 'starfish', 'stegosaurus', 'stop sign', 'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella', 'watch', 'water lilly', 'wheelchair', 'wild cat', 'windsor chair', 'wrench', 'yin and yang symbol'
    ],
    imagenet1k = [
        "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "rooster", 
        "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", 
        "jay", "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle", "vulture", "great grey owl",
        "fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", 
        "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", 
        "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", 
        "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", 
        "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", 
        "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper",
        "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider",
        "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse",
        "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", 
        "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", 
        "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", 
        "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", 
        "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", 
        "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot", "bustard", 
        "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", 
        "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", 
        "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", 
        "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", 
        "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", 
        "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", 
        "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", 
        "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", 
        "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", 
        "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", 
        "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", 
        "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", 
        "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie",
        "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", 
        "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", 
        "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", 
        "Newfoundland dog", "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon", "Pembroke Welsh Corgi", 
        "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", 
        "Alaskan tundra wolf", "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox",
        "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", 
        "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug",
        "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", 
        "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", 
        "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", 
        "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", 
        "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", 
        "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel",
        "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", 
        "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", 
        "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", 
        "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", 
        "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", 
        "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", 
        "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail", "barbell", 
        "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", 
        "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", "beer glass", "bell tower",
        "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase",
        "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", 
        "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", 
        "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel", "automated teller machine", "cassette", "cassette player", 
        "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "storage chest", 
        "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", 
        "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store", 
        "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", 
        "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", 
        "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", 
        "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", 
        "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute",
        "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", 
        "fur coat", "garbage truck", "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", 
        "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray", "half-track", "hammer", "hamper", 
        "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", 
        "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", 
        "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", 
        "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", 
        "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag", "mailbox",
        "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", 
        "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", 
        "missile", "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor", "moped", "mortar and pestle", 
        "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", 
        "metal nail", "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", 
        "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", 
        "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", 
        "railroad car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", 
        "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship", 
        "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", 
        "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "missile", 
        "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", 
        "rain barrel", "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver",
        "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker",
        "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver",
        "seat belt", "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", 
        "shower cap", "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", 
        "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar",
        "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", 
        "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch",
        "stupa", "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", 
        "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television",
        "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", 
        "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran",
        "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano",
        "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball",
        "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower",
        "whiskey jug", "whistle", "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", 
        "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign", "traffic light",
        "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette", "bagel", "pretzel", 
        "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", 
        "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon", 
        "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", 
        "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", 
        "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", 
        "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", 
        "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"
    ],
    clevr_count_all = [
        "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    ],
    clevr_closest_object_distance = [
        "very nearby",
        "nearby",
        "near",
        "",
        "distant",
        "very distant",
    ],
    mnist = [
       "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    ],
    svhn = [
        "zero", "one", "two", "three", "four",
        "five", "six", "seven", "eight", "nine",
    ],
    kitti_closest_vehicle_distance = [
        "a photo i took of a car on my left or right side.",
        "a photo i took with a car nearby.",
        "a photo i took with a car in the distance.",
        "a photo i took with no car.",
    ],
    dmlab = [
        "nearby apple/melon",
        "far apple/melon",
        "very far apple/melon",
        "nearby lemon",
        "far lemon",
        "very far lemon",
    ],
    pets = [
    'Abyssinian', 'American Bulldog', 'American Pit Bull Terrier', 'Basset Hound', 'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer', 'British Shorthair', 'Chihuahua', 
    'Egyptian Mau', 'English Cocker Spaniel', 'English Setter', 'German Shorthaired', 'Great Pyrenees', 'Havanese', 'Japanese Chin', 
    'Keeshond', 'Leonberger', 'Maine Coon', 'Miniature Pinscher', 'Newfoundland', 'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 
    'Russian Blue', 'Saint Bernard', 'Samoyed', 'Scottish Terrier', 'Shiba Inu', 'Siamese', 'Sphynx', 'Staffordshire Bull Terrier', 
    'Wheaten Terrier', 'Yorkshire Terrier'
    ],
    pcam = [
      "lymph node",
      "lymph node containing metastatic tumor tissue",
    ],
    diabetic_retinopathy = [
        "no diabetic retinopathy",
        "mild diabetic retinopathy",
        "moderate diabetic retinopathy",
        "severe diabetic retinopathy",
        "proliferative diabetic retinopathy"
    ],
)

# default template to use when the dataset name does not belong to `zeroshot_classification_templates`
DEFAULT_ZEROSHOT_CLASSIFICATION_TEMPLATES = zeroshot_classification_templates["imagenet1k"]

# use by imagenet robustness datasets
all_imagenet_wordnet_ids = ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041', 'n01514668', 'n01514859', 'n01518878', 'n01530575', 'n01531178', 'n01532829', 'n01534433', 'n01537544', 'n01558993', 'n01560419', 'n01580077', 'n01582220', 'n01592084', 'n01601694', 'n01608432', 'n01614925', 'n01616318', 'n01622779', 'n01629819', 'n01630670', 'n01631663', 'n01632458', 'n01632777', 'n01641577', 'n01644373', 'n01644900', 'n01664065', 'n01665541', 'n01667114', 'n01667778', 'n01669191', 'n01675722', 'n01677366', 'n01682714', 'n01685808', 'n01687978', 'n01688243', 'n01689811', 'n01692333', 'n01693334', 'n01694178', 'n01695060', 'n01697457', 'n01698640', 'n01704323', 'n01728572', 'n01728920', 'n01729322', 'n01729977', 'n01734418', 'n01735189', 'n01737021', 'n01739381', 'n01740131', 'n01742172', 'n01744401', 'n01748264', 'n01749939', 'n01751748', 'n01753488', 'n01755581', 'n01756291', 'n01768244', 'n01770081', 'n01770393', 'n01773157', 'n01773549', 'n01773797', 'n01774384', 'n01774750', 'n01775062', 'n01776313', 'n01784675', 'n01795545', 'n01796340', 'n01797886', 'n01798484', 'n01806143', 'n01806567', 'n01807496', 'n01817953', 'n01818515', 'n01819313', 'n01820546', 'n01824575', 'n01828970', 'n01829413', 'n01833805', 'n01843065', 'n01843383', 'n01847000', 'n01855032', 'n01855672', 'n01860187', 'n01871265', 'n01872401', 'n01873310', 'n01877812', 'n01882714', 'n01883070', 'n01910747', 'n01914609', 'n01917289', 'n01924916', 'n01930112', 'n01943899', 'n01944390', 'n01945685', 'n01950731', 'n01955084', 'n01968897', 'n01978287', 'n01978455', 'n01980166', 'n01981276', 'n01983481', 'n01984695', 'n01985128', 'n01986214', 'n01990800', 'n02002556', 'n02002724', 'n02006656', 'n02007558', 'n02009229', 'n02009912', 'n02011460', 'n02012849', 'n02013706', 'n02017213', 'n02018207', 'n02018795', 'n02025239', 'n02027492', 'n02028035', 'n02033041', 'n02037110', 'n02051845', 'n02056570', 'n02058221', 'n02066245', 'n02071294', 'n02074367', 'n02077923', 'n02085620', 'n02085782', 'n02085936', 'n02086079', 'n02086240', 'n02086646', 'n02086910', 'n02087046', 'n02087394', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02088632', 'n02089078', 'n02089867', 'n02089973', 'n02090379', 'n02090622', 'n02090721', 'n02091032', 'n02091134', 'n02091244', 'n02091467', 'n02091635', 'n02091831', 'n02092002', 'n02092339', 'n02093256', 'n02093428', 'n02093647', 'n02093754', 'n02093859', 'n02093991', 'n02094114', 'n02094258', 'n02094433', 'n02095314', 'n02095570', 'n02095889', 'n02096051', 'n02096177', 'n02096294', 'n02096437', 'n02096585', 'n02097047', 'n02097130', 'n02097209', 'n02097298', 'n02097474', 'n02097658', 'n02098105', 'n02098286', 'n02098413', 'n02099267', 'n02099429', 'n02099601', 'n02099712', 'n02099849', 'n02100236', 'n02100583', 'n02100735', 'n02100877', 'n02101006', 'n02101388', 'n02101556', 'n02102040', 'n02102177', 'n02102318', 'n02102480', 'n02102973', 'n02104029', 'n02104365', 'n02105056', 'n02105162', 'n02105251', 'n02105412', 'n02105505', 'n02105641', 'n02105855', 'n02106030', 'n02106166', 'n02106382', 'n02106550', 'n02106662', 'n02107142', 'n02107312', 'n02107574', 'n02107683', 'n02107908', 'n02108000', 'n02108089', 'n02108422', 'n02108551', 'n02108915', 'n02109047', 'n02109525', 'n02109961', 'n02110063', 'n02110185', 'n02110341', 'n02110627', 'n02110806', 'n02110958', 'n02111129', 'n02111277', 'n02111500', 'n02111889', 'n02112018', 'n02112137', 'n02112350', 'n02112706', 'n02113023', 'n02113186', 'n02113624', 'n02113712', 'n02113799', 'n02113978', 'n02114367', 'n02114548', 'n02114712', 'n02114855', 'n02115641', 'n02115913', 'n02116738', 'n02117135', 'n02119022', 'n02119789', 'n02120079', 'n02120505', 'n02123045', 'n02123159', 'n02123394', 'n02123597', 'n02124075', 'n02125311', 'n02127052', 'n02128385', 'n02128757', 'n02128925', 'n02129165', 'n02129604', 'n02130308', 'n02132136', 'n02133161', 'n02134084', 'n02134418', 'n02137549', 'n02138441', 'n02165105', 'n02165456', 'n02167151', 'n02168699', 'n02169497', 'n02172182', 'n02174001', 'n02177972', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02229544', 'n02231487', 'n02233338', 'n02236044', 'n02256656', 'n02259212', 'n02264363', 'n02268443', 'n02268853', 'n02276258', 'n02277742', 'n02279972', 'n02280649', 'n02281406', 'n02281787', 'n02317335', 'n02319095', 'n02321529', 'n02325366', 'n02326432', 'n02328150', 'n02342885', 'n02346627', 'n02356798', 'n02361337', 'n02363005', 'n02364673', 'n02389026', 'n02391049', 'n02395406', 'n02396427', 'n02397096', 'n02398521', 'n02403003', 'n02408429', 'n02410509', 'n02412080', 'n02415577', 'n02417914', 'n02422106', 'n02422699', 'n02423022', 'n02437312', 'n02437616', 'n02441942', 'n02442845', 'n02443114', 'n02443484', 'n02444819', 'n02445715', 'n02447366', 'n02454379', 'n02457408', 'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02483708', 'n02484975', 'n02486261', 'n02486410', 'n02487347', 'n02488291', 'n02488702', 'n02489166', 'n02490219', 'n02492035', 'n02492660', 'n02493509', 'n02493793', 'n02494079', 'n02497673', 'n02500267', 'n02504013', 'n02504458', 'n02509815', 'n02510455', 'n02514041', 'n02526121', 'n02536864', 'n02606052', 'n02607072', 'n02640242', 'n02641379', 'n02643566', 'n02655020', 'n02666196', 'n02667093', 'n02669723', 'n02672831', 'n02676566', 'n02687172', 'n02690373', 'n02692877', 'n02699494', 'n02701002', 'n02704792', 'n02708093', 'n02727426', 'n02730930', 'n02747177', 'n02749479', 'n02769748', 'n02776631', 'n02777292', 'n02782093', 'n02783161', 'n02786058', 'n02787622', 'n02788148', 'n02790996', 'n02791124', 'n02791270', 'n02793495', 'n02794156', 'n02795169', 'n02797295', 'n02799071', 'n02802426', 'n02804414', 'n02804610', 'n02807133', 'n02808304', 'n02808440', 'n02814533', 'n02814860', 'n02815834', 'n02817516', 'n02823428', 'n02823750', 'n02825657', 'n02834397', 'n02835271', 'n02837789', 'n02840245', 'n02841315', 'n02843684', 'n02859443', 'n02860847', 'n02865351', 'n02869837', 'n02870880', 'n02871525', 'n02877765', 'n02879718', 'n02883205', 'n02892201', 'n02892767', 'n02894605', 'n02895154', 'n02906734', 'n02909870', 'n02910353', 'n02916936', 'n02917067', 'n02927161', 'n02930766', 'n02939185', 'n02948072', 'n02950826', 'n02951358', 'n02951585', 'n02963159', 'n02965783', 'n02966193', 'n02966687', 'n02971356', 'n02974003', 'n02977058', 'n02978881', 'n02979186', 'n02980441', 'n02981792', 'n02988304', 'n02992211', 'n02992529', 'n02999410', 'n03000134', 'n03000247', 'n03000684', 'n03014705', 'n03016953', 'n03017168', 'n03018349', 'n03026506', 'n03028079', 'n03032252', 'n03041632', 'n03042490', 'n03045698', 'n03047690', 'n03062245', 'n03063599', 'n03063689', 'n03065424', 'n03075370', 'n03085013', 'n03089624', 'n03095699', 'n03100240', 'n03109150', 'n03110669', 'n03124043', 'n03124170', 'n03125729', 'n03126707', 'n03127747', 'n03127925', 'n03131574', 'n03133878', 'n03134739', 'n03141823', 'n03146219', 'n03160309', 'n03179701', 'n03180011', 'n03187595', 'n03188531', 'n03196217', 'n03197337', 'n03201208', 'n03207743', 'n03207941', 'n03208938', 'n03216828', 'n03218198', 'n03220513', 'n03223299', 'n03240683', 'n03249569', 'n03250847', 'n03255030', 'n03259280', 'n03271574', 'n03272010', 'n03272562', 'n03290653', 'n03291819', 'n03297495', 'n03314780', 'n03325584', 'n03337140', 'n03344393', 'n03345487', 'n03347037', 'n03355925', 'n03372029', 'n03376595', 'n03379051', 'n03384352', 'n03388043', 'n03388183', 'n03388549', 'n03393912', 'n03394916', 'n03400231', 'n03404251', 'n03417042', 'n03424325', 'n03425413', 'n03443371', 'n03444034', 'n03445777', 'n03445924', 'n03447447', 'n03447721', 'n03450230', 'n03452741', 'n03457902', 'n03459775', 'n03461385', 'n03467068', 'n03476684', 'n03476991', 'n03478589', 'n03481172', 'n03482405', 'n03483316', 'n03485407', 'n03485794', 'n03492542', 'n03494278', 'n03495258', 'n03496892', 'n03498962', 'n03527444', 'n03529860', 'n03530642', 'n03532672', 'n03534580', 'n03535780', 'n03538406', 'n03544143', 'n03584254', 'n03584829', 'n03590841', 'n03594734', 'n03594945', 'n03595614', 'n03598930', 'n03599486', 'n03602883', 'n03617480', 'n03623198', 'n03627232', 'n03630383', 'n03633091', 'n03637318', 'n03642806', 'n03649909', 'n03657121', 'n03658185', 'n03661043', 'n03662601', 'n03666591', 'n03670208', 'n03673027', 'n03676483', 'n03680355', 'n03690938', 'n03691459', 'n03692522', 'n03697007', 'n03706229', 'n03709823', 'n03710193', 'n03710637', 'n03710721', 'n03717622', 'n03720891', 'n03721384', 'n03724870', 'n03729826', 'n03733131', 'n03733281', 'n03733805', 'n03742115', 'n03743016', 'n03759954', 'n03761084', 'n03763968', 'n03764736', 'n03769881', 'n03770439', 'n03770679', 'n03773504', 'n03775071', 'n03775546', 'n03776460', 'n03777568', 'n03777754', 'n03781244', 'n03782006', 'n03785016', 'n03786901', 'n03787032', 'n03788195', 'n03788365', 'n03791053', 'n03792782', 'n03792972', 'n03793489', 'n03794056', 'n03796401', 'n03803284', 'n03804744', 'n03814639', 'n03814906', 'n03825788', 'n03832673', 'n03837869', 'n03838899', 'n03840681', 'n03841143', 'n03843555', 'n03854065', 'n03857828', 'n03866082', 'n03868242', 'n03868863', 'n03871628', 'n03873416', 'n03874293', 'n03874599', 'n03876231', 'n03877472', 'n03877845', 'n03884397', 'n03887697', 'n03888257', 'n03888605', 'n03891251', 'n03891332', 'n03895866', 'n03899768', 'n03902125', 'n03903868', 'n03908618', 'n03908714', 'n03916031', 'n03920288', 'n03924679', 'n03929660', 'n03929855', 'n03930313', 'n03930630', 'n03933933', 'n03935335', 'n03937543', 'n03938244', 'n03942813', 'n03944341', 'n03947888', 'n03950228', 'n03954731', 'n03956157', 'n03958227', 'n03961711', 'n03967562', 'n03970156', 'n03976467', 'n03976657', 'n03977966', 'n03980874', 'n03982430', 'n03983396', 'n03991062', 'n03992509', 'n03995372', 'n03998194', 'n04004767', 'n04005630', 'n04008634', 'n04009552', 'n04019541', 'n04023962', 'n04026417', 'n04033901', 'n04033995', 'n04037443', 'n04039381', 'n04040759', 'n04041544', 'n04044716', 'n04049303', 'n04065272', 'n04067472', 'n04069434', 'n04070727', 'n04074963', 'n04081281', 'n04086273', 'n04090263', 'n04099969', 'n04111531', 'n04116512', 'n04118538', 'n04118776', 'n04120489', 'n04125021', 'n04127249', 'n04131690', 'n04133789', 'n04136333', 'n04141076', 'n04141327', 'n04141975', 'n04146614', 'n04147183', 'n04149813', 'n04152593', 'n04153751', 'n04154565', 'n04162706', 'n04179913', 'n04192698', 'n04200800', 'n04201297', 'n04204238', 'n04204347', 'n04208210', 'n04209133', 'n04209239', 'n04228054', 'n04229816', 'n04235860', 'n04238763', 'n04239074', 'n04243546', 'n04251144', 'n04252077', 'n04252225', 'n04254120', 'n04254680', 'n04254777', 'n04258138', 'n04259630', 'n04263257', 'n04264628', 'n04265275', 'n04266014', 'n04270147', 'n04273569', 'n04275548', 'n04277352', 'n04285008', 'n04286575', 'n04296562', 'n04310018', 'n04311004', 'n04311174', 'n04317175', 'n04325704', 'n04326547', 'n04328186', 'n04330267', 'n04332243', 'n04335435', 'n04336792', 'n04344873', 'n04346328', 'n04347754', 'n04350905', 'n04355338', 'n04355933', 'n04356056', 'n04357314', 'n04366367', 'n04367480', 'n04370456', 'n04371430', 'n04371774', 'n04372370', 'n04376876', 'n04380533', 'n04389033', 'n04392985', 'n04398044', 'n04399382', 'n04404412', 'n04409515', 'n04417672', 'n04418357', 'n04423845', 'n04428191', 'n04429376', 'n04435653', 'n04442312', 'n04443257', 'n04447861', 'n04456115', 'n04458633', 'n04461696', 'n04462240', 'n04465501', 'n04467665', 'n04476259', 'n04479046', 'n04482393', 'n04483307', 'n04485082', 'n04486054', 'n04487081', 'n04487394', 'n04493381', 'n04501370', 'n04505470', 'n04507155', 'n04509417', 'n04515003', 'n04517823', 'n04522168', 'n04523525', 'n04525038', 'n04525305', 'n04532106', 'n04532670', 'n04536866', 'n04540053', 'n04542943', 'n04548280', 'n04548362', 'n04550184', 'n04552348', 'n04553703', 'n04554684', 'n04557648', 'n04560804', 'n04562935', 'n04579145', 'n04579432', 'n04584207', 'n04589890', 'n04590129', 'n04591157', 'n04591713', 'n04592741', 'n04596742', 'n04597913', 'n04599235', 'n04604644', 'n04606251', 'n04612504', 'n04613696', 'n06359193', 'n06596364', 'n06785654', 'n06794110', 'n06874185', 'n07248320', 'n07565083', 'n07579787', 'n07583066', 'n07584110', 'n07590611', 'n07613480', 'n07614500', 'n07615774', 'n07684084', 'n07693725', 'n07695742', 'n07697313', 'n07697537', 'n07711569', 'n07714571', 'n07714990', 'n07715103', 'n07716358', 'n07716906', 'n07717410', 'n07717556', 'n07718472', 'n07718747', 'n07720875', 'n07730033', 'n07734744', 'n07742313', 'n07745940', 'n07747607', 'n07749582', 'n07753113', 'n07753275', 'n07753592', 'n07754684', 'n07760859', 'n07768694', 'n07802026', 'n07831146', 'n07836838', 'n07860988', 'n07871810', 'n07873807', 'n07875152', 'n07880968', 'n07892512', 'n07920052', 'n07930864', 'n07932039', 'n09193705', 'n09229709', 'n09246464', 'n09256479', 'n09288635', 'n09332890', 'n09399592', 'n09421951', 'n09428293', 'n09468604', 'n09472597', 'n09835506', 'n10148035', 'n10565667', 'n11879895', 'n11939491', 'n12057211', 'n12144580', 'n12267677', 'n12620546', 'n12768682', 'n12985857', 'n12998815', 'n13037406', 'n13040303', 'n13044778', 'n13052670', 'n13054560', 'n13133613', 'n15075141']

# Official list of VTAB 19 tasks
VTAB_19TASKS = [
    "vtab/caltech101",
    "vtab/cifar100",
    "vtab/clevr_count_all",
    "vtab/clevr_closest_object_distance",
    "vtab/diabetic_retinopathy",
    "vtab/dmlab",
    "vtab/dsprites_label_orientation",
    "vtab/dsprites_label_x_position",
    "vtab/dtd",
    "vtab/eurosat",
    "vtab/kitti_closest_vehicle_distance",
    "vtab/flowers",
    "vtab/pets",
    "vtab/pcam",
    "vtab/resisc45",
    "vtab/smallnorb_label_azimuth",
    "vtab/smallnorb_label_elevation",
    "sun397",
    "vtab/svhn",
]
