import contextlib,io,json,os,hashlib
from pathlib import Path
import pandas as pd
from IPython.display import display
from sklearn.metrics import confusion_matrix,classification_report,precision_score,recall_score,average_precision_score
root=Path('/workspace/capstone-runs/threshold-test')
out=Path('/workspace/2026-capstone-design/results/threshold_relaxation')
# Freeze the exact candidate from the previously inspected training-only OOF results.
cv_path=Path('/workspace/capstone-runs/recall-validation/recall_thresholds.csv')
results=pd.read_csv(cv_path)
candidate=results[results.recall>=.50].sort_values(
    ['precision','recall','AP','top_n'],ascending=[False,False,False,True]
).iloc[0]
assert candidate['method']=='RF_weight'
assert int(candidate['top_n'])==50
assert abs(float(candidate['threshold'])-0.11752664641562667)<1e-14
(out/'selected_from_training.json').write_text(json.dumps(candidate.to_dict(),indent=2))
# Reuse the original pipeline, matching the pipeline that produced these OOF scores.
original=json.loads(Path('/workspace/capstone-runs/preprocessing-validation/before.ipynb').read_text())
os.chdir('/workspace/capstone-runs/recall-validation')
def without_shell(source):
    return '\n'.join(line for line in source.splitlines() if not line.startswith('!'))
setup=without_shell(''.join(original['cells'][1]['source']))
models=without_shell(''.join(original['cells'][3]['source']).split('for method in methods:')[0])
context={'display':display}
exec(setup,context)
exec(models,context)
model=context['make_model'](int(candidate['top_n']),candidate['method'])
model.fit(context['X_train'],context['y_train_binary'])
scores=model.predict_proba(context['X_test'])[:,1]
threshold=float(candidate['threshold'])
prediction=context['np'].where(scores>=threshold,1,-1)
y_test=context['y_test']
tn,fp,fn,tp=map(int,confusion_matrix(y_test,prediction,labels=[-1,1]).ravel())
record={
    'selection_rule':'From previous train-only OOF: recall >= 0.50, maximize precision; no further test tuning',
    'method':candidate['method'],'top_n':int(candidate['top_n']),'threshold':threshold,
    'cv_recall':float(candidate['recall']),'cv_precision':float(candidate['precision']),
    'test_size':len(y_test),'test_defects':int((y_test==1).sum()),
    'test_recall':recall_score(y_test,prediction,pos_label=1),
    'test_precision':precision_score(y_test,prediction,pos_label=1,zero_division=0),
    'AP':average_precision_score(y_test==1,scores),
    'tp':tp,'fn':fn,'fp':fp,'tn':tn,'false_positive_rate':fp/(tn+fp),
    'goal_50_percent_reached':tp/(tp+fn)>=.5,
    'cv_results_sha256':hashlib.sha256(cv_path.read_bytes()).hexdigest(),
}
assert tp+fn==21 and tn+fp==293
assert abs(record['test_recall']-tp/21)<1e-12
assert abs(record['test_precision']-tp/(tp+fp))<1e-12
(out/'metrics.json').write_text(json.dumps(record,indent=2))
pd.DataFrame({'row_index':y_test.index,'actual':y_test.to_numpy(),'score':scores,'predicted':prediction}).to_csv(out/'test_predictions.csv',index=False)
report=classification_report(y_test,prediction,labels=[-1,1],target_names=['normal','defect'],digits=4,zero_division=0)
(out/'classification_report.txt').write_text(report)
print(json.dumps(record,indent=2))
print(report)
