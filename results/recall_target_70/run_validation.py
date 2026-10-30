import hashlib,json,os,shutil
from pathlib import Path
import nbformat
from nbclient import NotebookClient
root=Path('/workspace/capstone-runs/recall-target-70')
out=Path('/workspace/2026-capstone-design/results/recall_target_70')
out.mkdir(exist_ok=True,parents=True)
os.chdir(root)
os.environ['PATH']='/workspace/capstone-venv/bin:'+os.environ['PATH']
os.environ['MPLBACKEND']='Agg'
shutil.copytree('/workspace/capstone-runs/preprocessing-validation/secom_data',root/'secom_data',dirs_exist_ok=True)
nb=nbformat.read('/workspace/2026-capstone-design/수정용_캡스톤디자인.ipynb',as_version=4)
previous=nbformat.read('/workspace/capstone-runs/preprocessing-validation/preprocessing.executed.ipynb',as_version=4)
def clean(s):return '\n'.join(l for l in s.splitlines() if not l.startswith('!'))
assert clean(nb.cells[1].source)==previous.cells[1].source
assert clean(nb.cells[3].source)==previous.cells[4].source
nb.cells[1].source=clean(nb.cells[1].source)
nb.cells[3].source=clean(nb.cells[3].source).split('for preprocessing, feature_counts in preprocessing_options.items():')[0]+'''
# The data split, model pipeline and threshold grid were verified unchanged.
# Reuse all 70 previously completed 5-Fold CV combinations, then refit selected models.
results_df = pd.read_csv('/workspace/capstone-runs/preprocessing-validation/cv_thresholds.csv')
assert results_df.groupby(['preprocessing','method','top_n']).ngroups == 70
print('Reused 70 validated 5-Fold CV combinations; no new model or preprocessing change.')
'''
# Persist the CV choice before evaluating the test data.
nb.cells[5].source+='''
from pathlib import Path
import json
out = Path('/workspace/2026-capstone-design/results/recall_target_70')
(out/'selected_before_test.json').write_text(json.dumps(best.to_dict(),indent=2))
'''
nb.cells.append(nbformat.v4.new_code_cell('''
from sklearn.metrics import confusion_matrix,classification_report
import hashlib, importlib.metadata as metadata
from IPython.display import display
cm = confusion_matrix(y_test,y_pred_final,labels=[-1,1])
tn,fp,fn,tp=map(int,cm.ravel())
record={
    'policy': {'cv_target_recall':target_recall,'cv_min_precision':min_precision,
               'ranking':'precision, recall, AP descending; no fallback'},
    'selected':best.to_dict(),'test_recall':float(test_recall),'test_precision':float(test_precision),
    'tp':tp,'fn':fn,'fp':fp,'tn':tn,'test_size':len(y_test),
    'test_defects':int((y_test==1).sum()),'false_positive_rate':fp/(tn+fp),
    'goal_above_50_reached':bool(test_recall>.50),
    'versions':{k:metadata.version(k) for k in ['scikit-learn','imbalanced-learn','numpy','pandas','xgboost']},
    'reused_cv_sha256':hashlib.sha256(Path('/workspace/capstone-runs/preprocessing-validation/cv_thresholds.csv').read_bytes()).hexdigest()
}
assert record['selected']['recall']>=.7 and record['selected']['precision']>=.1
assert tp+fn==21 and tn+fp==293
assert abs(test_recall-tp/(tp+fn))<1e-12
assert abs(test_precision-tp/(tp+fp))<1e-12
(out/'metrics.json').write_text(json.dumps(record,indent=2))
test_comparison.to_csv(out/'preprocessing_comparison.csv',index=False)
pd.DataFrame({'row_index':y_test.index,'actual':y_test.to_numpy(),'score':test_proba,'predicted':y_pred_final}).to_csv(out/'test_predictions.csv',index=False)
(out/'classification_report.txt').write_text(classification_report(y_test,y_pred_final,labels=[-1,1],target_names=['normal','defect'],digits=4,zero_division=0))
print(json.dumps(record,indent=2))
'''))
def start(cell,cell_index,**kwargs):
 if cell.cell_type=='code':print('Starting cell',cell_index,flush=True)
def done(cell,cell_index,**kwargs):
 if cell.cell_type=='code':
  print('Finished cell',cell_index,flush=True)
  for o in cell.get('outputs',[]):
   if o.output_type=='stream':print(o.text,flush=True)
client=NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(root)}},on_cell_start=start,on_cell_executed=done)
try:client.execute()
finally:nbformat.write(nb,str(out/'executed.ipynb'))
print('Validation complete',flush=True)
