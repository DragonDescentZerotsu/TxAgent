import pytest
from tools.chembl_tool.tasks.carcinogens.reviewed_conditions import classify_condition
from tools.chembl_tool.tasks.carcinogens.build_reviewed_starling import source_units

@pytest.mark.parametrize('model,scope,expected',[
 ('male F344 rat','experimental_animal','rodent'),
 ('B6C3F1 mice and hamsters','experimental_animal','rodent'),
 ('laboratory rodents','experimental_animal','rodent'),
 ('guinea pig (Cavia porcellus)','experimental_animal','rodent'),
 ('beagle (Canis lupus familiaris)','experimental_animal','dog'),
 ('Macaca mulatta','experimental_animal','monkey'),
 ('Oryctolagus cuniculus','experimental_animal','rabbit'),
 (None,'human','human'),('women workers','human','human'),
 ('humans and rats','mixed',None),('rat','human',None),
 ('zebrafish','experimental_animal',None),
 ('rat hepatocytes','in_vitro',None),('human cells','in_vitro',None),
 ('human oral epithelial cells','human',None),
 ('patients with T-cell acute lymphoblastic leukemia','human','human'),
 (None,'general_or_unspecified',None),('animals','experimental_animal',None),
 ('non-human primates','experimental_animal',None),
 ('non-human rhesus monkeys','experimental_animal','monkey'),
])
def test_organism_scope(model,scope,expected):
 assert classify_condition({'evidence_population_or_model':model,'evidence_scope':scope})['group']==expected

def test_recover_focal_outcome_without_model():
 r={'evidence_scope':'experimental_animal','support_text':'The compound caused liver tumors in mice.'}
 assert classify_condition(r)['group']=='rodent'
 r['support_text']='The compound caused tumors in mice, but no tumors in dogs.'
 assert classify_condition(r)['group'] is None

def test_model_controls_background_mentions():
 r={'evidence_scope':'experimental_animal','evidence_population_or_model':'rat','support_text':'Rats developed tumors. Relevance to humans is unknown.'}
 assert classify_condition(r)['group']=='rodent'

@pytest.mark.parametrize('support',[
 'Human exposure to the carcinogenic PAHs in soils may occur by ingestion.',
 'The kinetics explain carcinogenic effects and help assess risk to humans from acrylamide exposure.',
 'Lyngbyatoxin is a tumor promoter that stimulates activity in human polymorphonuclear leukocytes.',
])
def test_human_exposure_or_relevance_does_not_locate_tumor_outcome(support):
 assert classify_condition({'support_text':support})['group'] is None


def candidate(uid,y,pmid='1'):
 return {'source_record_id':uid,'Y':y,'study_id':'pmid:'+pmid,'pmid':pmid,'drug':'CC','bemis_murcko_scaffold':'',
         'molecule_identity_key':'P','condition_group':'species=rodent'}

def test_repeated_publication_is_one_vote():
 rs=[candidate('a',0),candidate('b',0)]
 votes,conflicts,_,disp=source_units(rs,{'a':{'support_text':'first'},'b':{'support_text':'second'}})
 assert len(votes)==1 and votes[0]['Y']==0 and not conflicts
 assert votes[0]['study_source_row_uids']==['a','b']
 assert disp['b']['disposition']=='duplicate_support_of_study_vote'

def test_within_publication_disagreement_is_not_two_votes():
 rs=[candidate('a',0),candidate('b',1)]
 votes,conflicts,_,_=source_units(rs,{'a':{},'b':{}})
 assert not votes and len(conflicts)==1

def test_exact_cross_publication_passage_deduplicates():
 text='This is an explicit source passage showing no tumors in the animals throughout the entire study.'
 rs=[candidate('a',0,'1'),candidate('b',0,'2')]
 votes,conflicts,links,_=source_units(rs,{'a':{'support_text':text},'b':{'support_text':text}})
 assert len(votes)==1 and len(links)==1 and not conflicts
