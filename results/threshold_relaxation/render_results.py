import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties, fontManager
out=Path('/workspace/2026-capstone-design/results/threshold_relaxation')
r=json.loads((out/'metrics.json').read_text())
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fontManager.addfont(font)
plt.rcParams.update({'font.family':FontProperties(fname=font).get_name(),'axes.unicode_minus':False})
fig=plt.figure(figsize=(12,8),facecolor='#f5f7fb')
fig.text(.06,.94,'불량 검출 성능 · 테스트 실행 결과',fontsize=23,weight='bold',color='#152238')
fig.text(.06,.898,f"가중치 적용 Random Forest  |  변수 {r['top_n']}개  |  판정 임계값 {r['threshold']:.8f}",fontsize=12,color='#536179')
cards=[(.06,'불량 검출률 (Recall)',f"{r['test_recall']:.2%}",f"실제 불량 {r['test_defects']}개 중 {r['tp']}개 검출",'#1769b2'),
       (.365,'불량 정밀도 (Precision)',f"{r['test_precision']:.2%}",f"불량 의심 {r['tp']+r['fp']}개 중 실제 불량 {r['tp']}개",'#6545a4'),
       (.67,'검출률 50% 목표','미달' if not r['goal_50_percent_reached'] else '달성',f"현재 {r['tp']}개 / 목표 11개 이상 검출",'#b33d2f')]
for x,title,value,note,color in cards:
 box=FancyBboxPatch((x,.66),.27,.19,boxstyle='round,pad=0.012',transform=fig.transFigure,facecolor='white',edgecolor='#e3e8f0',zorder=0)
 fig.patches.append(box)
 fig.text(x+.012,.805,title,fontsize=12,color='#536179')
 fig.text(x+.012,.731,value,fontsize=31,weight='bold',color=color)
 fig.text(x+.012,.680,note,fontsize=10,color='#536179')
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
fig.text(.55,.52,'교차검증과 테스트는 서로 다른 평가',fontsize=15,weight='bold',color='#152238')
fig.text(.55,.46,f"교차검증 검출률   {r['cv_recall']:.2%}",fontsize=14,color='#536179')
fig.text(.55,.41,f"테스트 검출률       {r['test_recall']:.2%}",fontsize=14,weight='bold',color='#1769b2')
fig.text(.55,.34,'검출률 = 9 ÷ (9 + 12) = 42.86%',fontsize=12,color='#152238')
fig.text(.55,.29,'정밀도 = 9 ÷ (9 + 38) = 19.15%',fontsize=12,color='#152238')
fig.text(.55,.21,'교차검증에서 정한 설정을 고정한 뒤\n테스트 데이터를 평가했습니다.',fontsize=11,color='#536179',linespacing=1.7)
fig.text(.06,.08,'SECOM · 층화 80:20 분할 · random_state=42 · 실측 평가값으로 생성한 결과 화면',fontsize=10,color='#536179')
fig.text(.06,.045,'동일한 테스트 분할을 이전 실험에서도 사용했습니다. 새로운 데이터의 성능을 보장하는 수치는 아닙니다.',fontsize=9,color='#6b778c')
fig.savefig(out/'test_results.png',dpi=160,facecolor=fig.get_facecolor())
fig.savefig(out/'test_results.pdf',facecolor=fig.get_facecolor())
print(out/'test_results.png')
