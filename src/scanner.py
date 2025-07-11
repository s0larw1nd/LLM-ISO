from langchain.text_splitter import TextSplitter
import re
from textwrap import wrap
from itertools import chain
    
class CustomTextSplitter(TextSplitter):
    def __init__(self, **kwargs):
        if 'chunk_size' not in kwargs: kwargs['chunk_overlap']=1000000
        kwargs['chunk_overlap']=0
        super().__init__(**kwargs)

    def split_text(self, text):
        text = re.sub(r"(?<!\d)\t", "", text)
        while '\n\n' in text: text = re.sub(r"\n\n", "\n", text)
        res = [chunk for chunk in re.split(r'\n(\d+(?:\.\d+)*(?: |\t|\n|    |\xa0)[\w .)(—,:\-«»]+\n[\s\S]+?)(?=\n\d+(?:\.\d+)*(?: |\t|\n|    |\xa0)[\w .)(—,:\-«»]+|\Z)', text) if len(chunk)>0]
        if hasattr(self,'_chunk_size'): res = list(chain(*[wrap(i,self._chunk_size) for i in res]))
        return res