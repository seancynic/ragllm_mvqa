import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from umlsparser import UMLSParser


class RelationParser:
    def __init__(self, path: str, save_path: str, datasrc: str, year: str):
        """
        :param path: Basepath to UMLS data files
        :param languages: List of languages with three-letter style language codes (if empty, no filtering will be applied)
        """
        self.paths = {
            'MRDOC': os.path.join(path, 'META', 'MRDOC.RRF'),
            'MRREL': os.path.join(path, 'META', 'MRREL.RRF')
        }
        self.save_path = save_path
        self.data_source = datasrc
        self.year = year

        Path(self.save_path).mkdir(parents=True, exist_ok=True)

        self.__parse_mrdoc__()
        # self.__parse_mrrel__()
        self.__parse_mrrela__()
        self.__parse_mrdef__(path)

    def __parse_mrdoc__(self):
        mrdoc_list = []

        for line in tqdm(open(self.paths['MRDOC'], encoding='utf-8'), desc='Parsing UMLS Abbr Values (MRDOC.RRF)'):
            line = line.split('|')
            if (line[0] == 'REL' or line[0] == 'RELA') and line[2] == 'expanded_form':
                data = {
                    'VALUE': line[1],
                    'EXPL': line[3]
                }
                mrdoc_list.append(data)

        umls_doc_df = pd.DataFrame(mrdoc_list)
        umls_doc_df.to_csv(rf'{self.save_path}/{self.data_source}_{self.year}_DOC.csv', index=False)

    def __parse_mrrel__(self):
        mrrel_list = []

        for line in tqdm(open(self.paths['MRREL'], encoding='utf-8'), desc='Parsing UMLS relations (MRREL.RRF)'):
            line = line.split('|')
            if line[0] != line[4]:
                # Get only REL
                if line[3] != '':
                    data = {
                        'CUI1': line[0],
                        'REL': line[3],
                        'CUI2': line[4]
                    }
                else:
                    continue
                mrrel_list.append(data)

        umls_rel_df = pd.DataFrame(mrrel_list)
        umls_rel_df.to_csv(rf'{self.save_path}/{self.data_source}_{self.year}_REL.csv', index=False)

    def __parse_mrrela__(self):
        mrrela_list = []

        for line in tqdm(open(self.paths['MRREL'], encoding='utf-8'), desc='Parsing UMLS relations (MRREL.RRF)'):
            line = line.split('|')
            if line[0] != line[4]:
                # Get only RELA
                if line[7] != '':
                    data = {
                        'CUI1': line[0],
                        'REL': line[7],
                        'CUI2': line[4]
                    }
                else:
                    continue
                mrrela_list.append(data)

        umls_rel_df = pd.DataFrame(mrrela_list)
        umls_rel_df.to_csv(rf'{self.save_path}/{self.data_source}_{self.year}_RELA.csv', index=False)

    def __parse_mrdef__(self, path):
        umls_def_list = []
        umls = UMLSParser(path)

        for cui, concept in umls.get_concepts().items():
            # get the definition
            defs = concept.get_definitions()
            umls_def_list.append({
                'cui': cui,
                'name': concept.get_names_for_language('ENG')[0],
                'definition': '' if len(defs) == 0 else defs.pop()[0],
            })

        umls_def_df = pd.DataFrame(umls_def_list)
        umls_def_df.to_csv(rf'{self.save_path}/{self.data_source}_{self.year}_DEF.csv', index=False)


if __name__ == '__main__':
    RelationParser('/2024AB', 'umls_datasets', 'UMLS', '2024AB')