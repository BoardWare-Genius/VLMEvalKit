import warnings

from .image_base import ImageBaseDataset
from .image_mcq import ImageMCQDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *

class TourMCQ(ImageMCQDataset):
    TYPE = 'MCQ'

    DATASET_URL = {
                # Lanchao
                #'TOUR-MO-V': '/iivanwu/data/LMUData/test/TourMO_V_MCQ.json',
                #'TOUR-MO-ENT': '/iivanwu/data/LMUData/test/TourMO_Entity_MCQ.json',
                # Zstack
                'TOUR-MO-V': '/Dataset/Domain/LMUData/test/TourMO_V_MCQ.json',
                'TOUR-MO-ENT': '/Dataset/Domain/LMUData/test/TourMO_Entity_MCQ.json',
                }

    DATASET_MD5 = {}

    # def evaluate(self, eval_file, **judge_kwargs):
    #     from .utils.multiple_choice import report_acc, report_acc_MMT, mcq_circular_eval, mcq_vanilla_eval
    def build_prompt(self, line):

        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above. \
            The answer only need to contain the correct answer option without any explanation of redundancy \n'

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs
        
    def load_data(self, dataset):
        data_path = self.DATASET_URL[dataset]
        tsv_file = data_path.replace('.json','.tsv')
        # file_md5 = self.DATASET_MD5[dataset] if dataset in self.DATASET_MD5 else None
        data = pd.DataFrame(load(data_path))
        # data.reset_index(inplace=True,drop=True)
        # data['index'] = data.index
        path_prefix = osp.dirname(data_path)
        data['image_path'] = data['image_path'].apply(lambda x: x.replace('./',path_prefix+'/'))
        # data['image_path'] = data['image_path'].apply(lambda x: osp.join(path_prefix,x))
        data.to_csv(tsv_file, sep='\t',index=False)
        update_flag = False
        if file_size(data_path, 'GB') > 1:
            local_path = tsv_file.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None) or update_flag:
                from vlmeval.tools import LOCALIZE
                LOCALIZE(data_path, local_path)
            data_path = local_path
        # import pdb; pdb.set_trace()
        return data
