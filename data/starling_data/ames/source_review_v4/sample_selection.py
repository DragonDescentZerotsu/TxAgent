import json,hashlib,re,sys
from pathlib import Path
from collections import defaultdict
import pyarrow.parquet as pq
R=Path('data/starling_data/ames'); O=Path(sys.argv[1]); O.mkdir(parents=True,exist_ok=True)
used=set(json.loads((R/'source_review_v4/previous_sampled_ids.json').read_text()))
votes=[json.loads(l) for l in (R/'source_review_v4/previous_source_votes.jsonl').read_text().splitlines()]
audit=pq.read_table(R/'source_review_v4/sampling_L2_frame.parquet').to_pandas()
rank=lambda rid:hashlib.sha256(('two-sided-20260906:'+rid).encode()).hexdigest()
selected=[]
base=pq.read_table(R/'raw_v1/ames_base.parquet')
rows=base.take([int(v['source_record_id'].split(':')[1]) for v in votes]).to_pylist()
buckets=defaultdict(list)
for v,raw in zip(votes,rows):
 rid=v['source_record_id']
 if rid in used: continue
 act=raw['metabolic_activation']
 x=dict(id=rid,raw=raw,vote=v,current_group='Direct.ames',identity_status='verified_parent_match',parent_inchi_key=v['molecule_identity_key'],rank=rank(rid))
 buckets[(v['Y'],act)].append(x)
for key,pool in sorted(buckets.items()):
 pmids=set()
 for x in sorted(pool,key=lambda x:x['rank']):
  if x['raw']['pmid'] in pmids:continue
  x['stratum']='L1_stratified_'+str(key);selected.append(x);used.add(x['id']);pmids.add(x['raw']['pmid'])
  if len(pmids)==12:break
# 72 stratified +24 risk-prioritized, no automatic verdict.
risks=[]
for pool in buckets.values():
 for x in pool:
  if x['id'] in used:continue
  body=str(x['raw']['support_text'])+' '+str(x['raw'].get('extra_details'))
  terms=re.findall(r'review|summari[sz]|literature|control|not.{0,20}signific|slight|weak|slope|toxic|negative|other|respectively|indirect|analogue|derivatives',body,re.I)
  x['priority_score']=len(set(t.lower() for t in terms));risks.append(x)
pmids=set()
for x in sorted(risks,key=lambda x:(-x['priority_score'],x['rank'])):
 if x['raw']['pmid'] in pmids:continue
 x['stratum']='L1_targeted_semantic_risk';selected.append(x);used.add(x['id']);pmids.add(x['raw']['pmid'])
 if len(pmids)==24:break
for source in ['ames_base','ames_v1','ames_v2','ames_v3']:
 meta=audit[(audit.source_id==source)&(audit.group_id=='Observed.nonvoter_ames')&~audit.source_record_id.isin(used)]
 table=pq.read_table(R/'raw_v1'/f'{source}.parquet').take(meta.source_row_number.tolist())
 pool=[]
 for raw,m in zip(table.to_pylist(),meta.to_dict('records')):
  body=str(raw.get('support_text',''))+' '+str(raw.get('extra_details',''))
  if not re.search(r'\bTA[ -]?\d|\bWP2|\bames\b',body,re.I):continue
  if not re.search('mutagen|revertant|reverse.mutation',body,re.I):continue
  score=3*bool(re.search('revertant',body,re.I))+2*bool(re.search(r'S[ -]?9|without.{0,25}activation|with.{0,25}activation',body,re.I))+bool(re.search(r'\bTA[ -]?\d|\bWP2',body,re.I))+bool(re.search('positive|negative|non.mutagen|mutagenic|dose',body,re.I))
  if source=='ames_base':score+=2*(m['identity_verification']=='verified_parent_match')
  pool.append(dict(id=m['source_record_id'],raw=raw,stratum=source+'_recall_targeted',current_group=m['group_id'],identity_status=m['identity_verification'],parent_inchi_key=m['parent_inchi_key'],final_reason=m['reason'],priority_score=score,rank=rank(m['source_record_id'])))
 pmids=set()
 for x in sorted(pool,key=lambda x:(-x['priority_score'],x['rank'])):
  if x['raw']['pmid'] in pmids:continue
  selected.append(x);pmids.add(x['raw']['pmid'])
  if len(pmids)==12:break
for i,x in enumerate(selected,1):x['review_number']=i
(O/'sample.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in selected))
print('sample',len(selected),'L1',sum(x['current_group']=='Direct.ames' for x in selected))
