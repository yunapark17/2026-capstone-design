import json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/capstone-mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties,fontManager
out=Path('/workspace/2026-capstone-design/results/recall_target_70')
r=json.loads((out/'metrics.json').read_text())
previous=json.loads(Path('/workspace/2026-capstone-design/results/threshold_relaxation/metrics.json').read_text())
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fontManager.addfont(font)
plt.rcParams.update({'font.family':FontProperties(fname=font).get_name(),'axes.unicode_minus':False})
fig=plt.figure(figsize=(12,8),facecolor='#f5f7fb')
fig.text(.06,.94,'불량 의심 범위 확대 · 실제 테스트 결과',fontsize=23,weight='bold',color='#152238')
fig.text(.06,.897,'가중치 적용 Random Forest · 전체 변수 사용 · 판정 임계값 0.07',fontsize=12,color='#536179')
cards=[(.06,'불량 검출률 (Recall)',f"{r['test_recall']:.2%}",f"불량 {r['test_defects']}개 중 {r['tp']}개 검출 · 50% 초과",'#167353'),
       (.365,'불량 정밀도 (Precision)',f"{r['test_precision']:.2%}",f"불량 의심 {r['tp']+r['fp']}개 중 실제 불량 {r['tp']}개",'#6545a4'),
       (.67,'정상 제품 오탐',f"{r['fp']}개",f"정상 {r['tn']+r['fp']}개 중 {r['false_positive_rate']:.2%}",'#b35d17')]
for x,title,value,note,color in cards:
 fig.patches.append(FancyBboxPatch((x,.66),.27,.19,boxstyle='round,pad=0.012',transform=fig.transFigure,facecolor='white',edgecolor='#e3e8f0',zorder=0))
 fig.text(x+.012,.805,title,fontsize=12,color='#536179')
 fig.text(x+.012,.731,value,fontsize=31,weight='bold',color=color)
 fig.text(x+.012,.68,note,fontsize=10,color='#536179')
fig.text(.06,.595,'테스트 판정 내역',fontsize=16,weight='bold',color='#152238')
fig.text(.06,.56,'평가 표본 314개 = 정상 293개 + 불량 21개',fontsize=11,color='#536179')
ax=fig.add_axes([.12,.20,.33,.30])
values=[[r['tn'],r['fp']],[r['fn'],r['tp']]]
colors=[['#e6eef8','#fce9dc'],['#fce9dc','#ddf0e6']]
for row in range(2):
 for col in range(2):
  ax.add_patch(plt.Rectangle((col,row),1,1,facecolor=colors[row][col],edgecolor='white',linewidth=3))
  name=[['정상 판정','정상 오탐'],['놓친 불량','불량 검출']][row][col]
  ax.text(col+.5,row+.41,f'{values[row][col]}개',ha='center',va='center',fontsize=24,weight='bold',color='#152238')
  ax.text(col+.5,row+.73,name,ha='center',va='center',fontsize=11,color='#536179')
ax.set(xlim=(0,2),ylim=(2,0),xticks=[.5,1.5],yticks=[.5,1.5])
ax.set_xticklabels(['예측: 정상','예측: 불량'],fontsize=11)
ax.set_yticklabels(['실제 정상','실제 불량'],fontsize=11)
ax.tick_params(length=0,pad=10)
for spine in ax.spines.values():spine.set_visible(False)
fig.text(.55,.525,'직전 평가 → 이번 평가',fontsize=16,weight='bold',color='#152238')
fig.text(.55,.465,f"검출률   {previous['test_recall']:.2%}  →  {r['test_recall']:.2%}",fontsize=15,weight='bold',color='#167353')
fig.text(.55,.405,f"정밀도   {previous['test_precision']:.2%}  →  {r['test_precision']:.2%}",fontsize=15,color='#6545a4')
fig.text(.55,.345,f"정상 오탐   {previous['fp']}개  →  {r['fp']}개",fontsize=15,weight='bold',color='#b35d17')
fig.text(.55,.26,'교차검증 검출률 70% 이상·정밀도 10% 이상\n후보 중 정밀도가 가장 높은 설정을 먼저 선택',fontsize=11,color='#536179',linespacing=1.7)
fig.text(.55,.18,'선택된 후보의 교차검증 결과\n검출률 73.49% · 정밀도 11.05%',fontsize=11,color='#536179',linespacing=1.7)
fig.text(.06,.08,'목표 검출률은 넘었지만, 불량 의심 129개 중 112개는 정상입니다. 재검사 부담이 커졌습니다.',fontsize=11,color='#9b4a16')
fig.text(.06,.04,'동일한 테스트 분할을 반복 사용한 비교 결과입니다. 새로운 데이터에서의 검출률을 보장하지 않습니다.',fontsize=9,color='#6b778c')
fig.savefig(out/'test_results.png',dpi=160,facecolor=fig.get_facecolor())
fig.savefig(out/'test_results.pdf',facecolor=fig.get_facecolor())
print(out/'test_results.png')
