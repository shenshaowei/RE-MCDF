from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import re

class NER_Model:
    def __init__(self, ner_model_id, device='gpu'):
        # 直接使用 pipeline，不传 preprocessor
        self.ner_model = pipeline(
            task=Tasks.named_entity_recognition,
            model=ner_model_id,
            device='gpu',  
        )

    def ner(self, text):
        if len(text) > 512 - 2:
            text = text[:510]  # 保留 [CLS] 和 [SEP] 位置
        try:
            result = self.ner_model([text], batch_size = 4)
        except:
            result = self.ner_model([t for t in text.split("\n") if t != ""], batch_size = 4)
        
        return result
    
# TO DO: Try more ner models