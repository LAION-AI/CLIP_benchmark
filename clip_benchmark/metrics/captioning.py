import json
from open_clip import tokenize
from tqdm.auto import tqdm
from open_clip.tokenizer import _tokenizer


from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


"""
Code adapted from https://github.com/salaniz/pycocoevalcap/blob/master/eval.py
Thanks to @salaniz for the code!
"""
class COCOEvalCap:
    def __init__(self, results):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.results = results
    def evaluate(self):
        gts = {}
        res = {}
        for imgId, r in enumerate(self.results):
            gts[imgId] = r['true']
            res[imgId] = r['gen']
        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]

def evaluate(model, dataloader, batch_size, device, transform, train_dataloader=None, num_workers=None, amp=True, verbose=False):
    results = []
    image_id = 0
    gt = []
    for idx, (img, captions) in enumerate(tqdm(dataloader)):
        out = model.generate(img.to(device))
        decoded = [_tokenizer.decode(i).split("<end_of_text>")[0].replace("<start_of_text>", "").strip() for i in out.cpu().numpy()]
        for pred, true in zip(decoded, captions):
            true = [{'caption': t} for t in true]
            pred = [{'caption': pred}]
            results.append({"image_id":image_id, "gen":pred, "true": true})
            image_id += 1
    coco_eval = COCOEvalCap(results)
    coco_eval.evaluate()
    metrics = coco_eval.eval
    # print output evaluation scores
    for metric, score in metrics.items():
        print(f'{metric}: {score:.3f}')
    return metrics
