import json,hashlib,collections,shutil,re
from pathlib import Path
import pyarrow.parquet as pq
import argparse
parser=argparse.ArgumentParser(description="Replay the 800-candidate pool and 200 reviewed selections into a separate directory.")
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
R=Path('data/starling_data/ames');D=R/'source_review_v5';O=args.output
if O.resolve()==D.resolve():raise ValueError('Replay output must be separate from the frozen audit')
O.mkdir(parents=True,exist_ok=True)
assert hashlib.sha256((R/'canonical_v1/manifest.json').read_bytes()).hexdigest()==json.loads((D/'summary.json').read_text())['input_lineage']['source_manifest_sha256'], 'Canonical source changed since sampling'
used=set(json.loads((D/'prior_reviewed_source_ids.json').read_text()))
columns=['source_record_id','source_id','group_id','canonical_endpoint_name','canonical_assay_context','canonical_measurement_text','support_text','molecule_identity_key','canonical_smiles']
rows=pq.read_table(R/'canonical_v1/records.parquet',columns=columns).to_pylist()
level={'Direct.ames':1,'Observed.nonvoter_ames':2,'Mechanism.other_genetic_damage':3,'Mechanism.dna_damage_response':4,'Mechanism.genotoxicity_mechanisms':5}
counts=collections.Counter((level[r['group_id']],r['source_id'],r['canonical_endpoint_name']) for r in rows)
(O/'endpoint_counts.json').write_text(json.dumps([{'level':l,'source':s,'endpoint':e,'count':n} for (l,s,e),n in sorted(counts.items())],ensure_ascii=False,indent=2))
pools=collections.defaultdict(list)
for r in rows:
 if r['source_record_id'] in used:continue
 r['level']=level[r['group_id']];r['rank']=hashlib.sha256(('level-semantic-20260906:'+r['source_record_id']).encode()).hexdigest();pools[(r['level'],r['source_id'])].append(r)
selected=[];chosen=set();quotas={(1,'ames_base'):80,(2,'ames_base'):100,(2,'ames_v1'):40,(2,'ames_v2'):50,(2,'ames_v3'):50,(3,'ames_base'):80,(3,'ames_v1'):80,(4,'ames_v2'):160,(5,'ames_v3'):160}
for key,q in quotas.items():
 by=collections.defaultdict(list)
 for r in sorted(pools[key],key=lambda r:r['rank']):by[r['canonical_endpoint_name']].append(r)
 n=0
 # Half stratified by native endpoint, then targeted family-boundary content.
 while n<q//2:
  changed=False
  for e in sorted(by):
   while by[e] and by[e][0]['source_record_id'] in chosen:by[e].pop(0)
   if not by[e]:continue
   r=by[e].pop(0);r['selection']='endpoint_stratified';selected.append(r);chosen.add(r['source_record_id']);n+=1;changed=True
   if n==q//2:break
  if not changed:break
 def risk(r):
  body=r['support_text']+' '+r['canonical_assay_context'];l=r['level']
  if l==1:return sum(bool(re.search(p,body,re.I)) for p in ['predict|qsar','repair|oxidative|comet','protect|inhibit','weak|equivocal'])
  if l==2:return 3*bool(re.search('revertant',body,re.I))+2*bool(re.search(r'S[ -]?9',body,re.I))+bool(re.search('positive|negative|mutagenic',body,re.I))+bool(re.search('control|weak|possibly|preincubation|whereas',body,re.I))
  if l==3:return 3*bool(re.search('comet|strand.break|DNA.adduct|gamma.H2AX|DNA.repair',body,re.I))+bool(re.search('ROS|oxidative|expression',body,re.I))
  if l==4:return 3*bool(re.search('micronucle|chromosom|HPRT|mutant.frequency|mutation.frequency',body,re.I))+bool(re.search('CYP|glutathione|ROS|antioxidant',body,re.I))
  return 3*bool(re.search('comet|strand.break|DNA.adduct|gamma.H2AX|DNA.damage|micronucle|chromosom|HPRT|mutation.frequency',body,re.I))+bool(re.search('prediction|predicted',body,re.I))
 for r in sorted(pools[key],key=lambda r:(-risk(r),r['rank'])):
  if r['source_record_id'] in chosen:continue
  r['selection']='boundary_or_recall_targeted';selected.append(r);chosen.add(r['source_record_id']);n+=1
  if n==q:break
 assert n==q,(key,n,q)
# Get only selected full raw records; no external paper data joins.
raw={}
for src in ['ames_base','ames_v1','ames_v2','ames_v3']:
 ss=[r for r in selected if r['source_id']==src];ords=[int(r['source_record_id'].split(':')[1]) for r in ss]
 for r,x in zip(ss,pq.read_table(R/'raw_v1'/f'{src}.parquet').take(ords).to_pylist()):raw[r['source_record_id']]=x
for i,r in enumerate(selected,1):r['review_number']=i;r['raw']=raw[r['source_record_id']]
(O/'sample_pool.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
reviewed=set(json.loads((D/'reviewed_numbers.json').read_text()))
reviewed_rows=[r for r in selected if r['review_number'] in reviewed]
assert len(reviewed_rows)==200
(O/'sample.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in reviewed_rows))
assert (O/'sample.jsonl').read_bytes()==(D/'sample.jsonl').read_bytes(), 'Sample replay differs'
print('REPLAY_OK',len(selected),'candidates;',len(reviewed_rows),'reviewed rows byte-identical',flush=True)
