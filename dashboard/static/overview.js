const displayNumber = new Intl.NumberFormat('pt-BR', {maximumFractionDigits:1});
const displayValue = value => Number.isFinite(value) ? displayNumber.format(value) : '—';
const put = (id, value) => { document.getElementById(id).textContent=value; };

function overviewLine(canvas, labels, datasets, unit){
  const draw=()=>{
    Chart.getChart(canvas)?.destroy();
    new Chart(canvas,{type:'line',data:{labels,datasets},options:{
      responsive:true,maintainAspectRatio:false,animation:false,
      interaction:{mode:'index',intersect:false},
      scales:{x:{ticks:{color:'#a2aec0',maxTicksLimit:8},grid:{display:false}},
        y:{beginAtZero:true,ticks:{color:'#a2aec0',callback:value=>value+(unit?' '+unit:'')},grid:{color:'#243044'}}},
      plugins:{legend:{display:false},tooltip:{
        displayColors:false,
        filter:item=>Boolean(unit) || ['Casos','Previsão','Máx','Mín'].includes(item.dataset.label)
          && (item.dataset.label==='Casos' || datasets[0].data[item.dataIndex]==null),
        itemSort:(a,b)=>['Máx','Previsão','Mín'].indexOf(a.dataset.label)-['Máx','Previsão','Mín'].indexOf(b.dataset.label),
        callbacks:{
          title:items=>unit?'':items[0]?.label.replace('Hoje · ','')||'',
          label:item=>unit?displayValue(item.parsed.y)+(unit==='%'?'':' ')+unit:
            (item.dataset.label==='Previsão'?'Méd':item.dataset.label)+': '+displayValue(item.parsed.y)
        }
      }},
      elements:{line:{tension:.25},point:{radius:2}}
    }});
  };
  if(window.Chart)draw();
  else document.getElementById('chart-library').addEventListener('load',()=>{if(canvas.isConnected)draw();},{once:true});
}

const shortWeek=week=>week?'S'+String(week).slice(-2):'—';

function renderClimate(weeks){
  const measures=[['temp_mean_mean','Temperatura média','°C'],['precip_total','Chuva acumulada','mm'],['humidity_mean','Umidade média','%']];
  const last=weeks.at(-1);
  put('climate-note','Histórico de 3 meses');
  const cards=measures.map(([key,label,unit])=>{
    const card=document.createElement('div');
    const title=document.createElement('span');
    const latest=document.createElement('strong');
    title.textContent=label;
    latest.textContent=displayValue(last[key])+' '+unit;
    const plot=document.createElement('div');
    plot.className='climate-plot';
    const canvas=document.createElement('canvas');
    canvas.setAttribute('role','img');
    canvas.setAttribute('aria-label',label+' nas últimas 12 semanas');
    plot.append(canvas);
    requestAnimationFrame(()=>{
      if(!canvas.isConnected)return;
      const color={temp_mean_mean:'#fbad58',precip_total:'#38bdf8',humidity_mean:'#2dd4bf'}[key];
      overviewLine(canvas,weeks.map(row=>shortWeek(row.SE)),[
        {label,data:weeks.map(row=>row[key]),borderColor:color,backgroundColor:color+'24',fill:true}
      ],unit);
    });
    const comparison=document.createElement('small');
    const previous=weeks.at(-2)?.[key];
    if(Number.isFinite(last[key]) && Number.isFinite(previous)){
      const change=Math.round((last[key]-previous)*10)/10;
      const deltaUnit=unit==='%'?'p.p.':unit;
      comparison.textContent=change===0?'Estável ante a semana anterior':
        (change>0?'Subiu ':'Caiu ')+displayValue(Math.abs(change))+' '+deltaUnit+' na semana';
    }else comparison.textContent='Comparação semanal indisponível';
    card.append(title,latest,comparison,plot);
    return card;
  });
  document.getElementById('climate-trends').replaceChildren(...cards);
}

let overviewRevision=0;
let municipalReference=null;
async function loadOverview(){
  const rev=++overviewRevision;
  const trend=document.getElementById('forecast-change');
  trend.hidden=true;
  try{
    const {weeks}=await json('/api/overview');
    if(rev!==overviewRevision)return;
    if(!weeks.length)throw Error('empty');
    renderClimate(weeks);
    const recent=weeks.slice(-8);
    const last=recent.at(-1);
    municipalReference=last;
    renderTrainingSubtitle();
    if(state.chart)renderChart(false);
    const past=recent.map(row=>({week:row.SE,value:row.casos}));
    const observed='Semana '+String(last.SE).slice(-2)+' · '+displayValue(last.casos)+' casos.';
    put('case-note',observed);
    let forecast;
    try{forecast=await loadForecast(last.SE);}catch{forecast=null;}
    if(rev!==overviewRevision)return;
    const rows=String(forecast?.origin_se)===String(last.SE) && ['issued','live'].includes(forecast.source)?forecast.forecast||[]:[];
    const future=Array.from({length:8},(_,i)=>{
      const row=rows.find(r=>Number(r.h)===i+1 && r.disponivel!==false);
      return {week:row?.target_se,value:row?.q50,low:row?.q05,high:row?.q95,innerLow:row?.q25,innerHigh:row?.q75};
    });
    showCitySummary(cityCode,last,future[7].value);
    const color=forecastColor(last,future[7].value);
    const history=past.map(p=>p.value);
    const empty=past.slice(0,-1).map(()=>null);
    const forecastSeries=key=>[
      ...empty,
      Number.isFinite(future[0][key]) ? history.at(-1) : null,
      ...future.map(point=>point[key])
    ];
    const labels=[...past.map((p,i)=>i===past.length-1?'Hoje · '+shortWeek(p.week):shortWeek(p.week)),
       ...future.map((p,i)=>p.week?shortWeek(p.week):'H'+(i+1))];
    const datasets=[{label:'Casos',data:[...history,...future.map(()=>null)],borderColor:'#2dd4bf'},
       {label:'Mín',data:forecastSeries('low'),borderWidth:0,pointRadius:0},
       {label:'Máx',data:forecastSeries('high'),borderWidth:0,pointRadius:0,backgroundColor:color+'26',fill:'-1'},
       {label:'Limite inferior 50%',data:forecastSeries('innerLow'),borderWidth:0,pointRadius:0},
       {label:'Limite superior 50%',data:forecastSeries('innerHigh'),borderWidth:0,pointRadius:0,backgroundColor:color+'47',fill:'-1'},
       {label:'Previsão',data:forecastSeries('value'),borderColor:color,borderDash:[5,4]}];
    renderCaseLegend('overview-legend',datasets,[
      [[0,'Casos']],[[5,'Previsão'],[4,'Faixa 50%'],[2,'Faixa 90%']]
    ]);
    overviewLine(document.getElementById('overview-chart'),labels,datasets,'');
    const end=future[7].value;
    if(Number.isFinite(last.casos) && Number.isFinite(end)){
      const delta=end-last.casos;
      const change=last.casos>0 ? (delta>0?'Aumento de ':delta<0?'Declínio de ':'Variação de ')+displayValue(Math.abs(100*delta/last.casos))+'%' :
        delta===0?'Sem variação':'Variação percentual indefinida (base zero)';
      const currentValue=document.createElement('strong');
      const forecastValue=document.createElement('strong');
      currentValue.textContent=displayValue(last.casos);
      forecastValue.textContent=displayValue(end);
      forecastValue.style.color=color;
      document.getElementById('case-note').replaceChildren(currentValue,' casos hoje e previsão de ',forecastValue,' casos daqui a 8 semanas');

      trend.textContent=(delta>0?'↑ ':delta<0?'↓ ':'')+change;
      trend.hidden=false;
    }else put('case-note',observed+' Previsão para 8 semanas indisponível.');

  }catch{
    if(rev!==overviewRevision)return;
    document.getElementById('climate-trends').replaceChildren();
    window.Chart?.getChart('overview-chart')?.destroy();
    put('case-note','Série municipal indisponível.'); put('climate-note','Série climática indisponível.');
  }
}

function showCitySummary(code,last,next){
  const link=document.getElementById('city-'+code);
  if(link.dataset.trend && !Number.isFinite(next))return;
  link.querySelector('span').textContent=last?shortWeek(last.SE)+' · '+displayValue(last.casos)+' casos':'Dados indisponíveis';
  let trend='Previsão indisponível';
  if(Number.isFinite(last?.casos) && Number.isFinite(next)){
    const delta=next-last.casos;
    trend=last.casos>0 ? (delta>0?'↑ ':delta<0?'↓ ':'')+displayValue(Math.abs(100*delta/last.casos))+'%':
      displayValue(last.casos)+' → '+displayValue(next)+' casos';
    trend+=' em 8 semanas';
  }
  link.querySelector('small').textContent=trend;
  if(Number.isFinite(last?.casos) && Number.isFinite(next)){
    const color=forecastColor(last,next);
    link.dataset.trend='forecast';
    link.style.setProperty('--city-trend',color);
    link.style.setProperty('--city-surface','color-mix(in srgb,'+color+' 20%, var(--panel))');
    link.style.setProperty('--city-border','color-mix(in srgb,'+color+' 60%, var(--line))');
    try{sessionStorage.setItem('episense-city-threshold-'+code,JSON.stringify({last,next}));}catch{}
  }
}

function restoreCitySummaries(){
  for(const code of cityCodes){
    try{
      const stored=JSON.parse(sessionStorage.getItem('episense-city-threshold-'+code));
      if(stored)showCitySummary(code,stored.last,stored.next);
    }catch{}
  }
}

async function loadOtherCity(code){
  const prefix=prefixForCity(code);
  let last;
  try{
    const overview=await json('/api/overview',prefix);
    last=overview.weeks.at(-1);
    showCitySummary(code,last);
    if(!last)return;
    const forecast=await json('/api/forecast?origin_se='+last.SE,prefix);
    const h8=forecast.forecast?.find(row=>Number(row.h)===8 && row.disponivel!==false);
    showCitySummary(code,last,['live','issued'].includes(forecast.source)?h8?.q50:null);
  }catch{showCitySummary(code,last);}
}

function calendarWeek(date){
  const [year,month,day]=new Intl.DateTimeFormat('sv-SE',{
    timeZone:'America/Campo_Grande',year:'numeric',month:'2-digit',day:'2-digit'
  }).format(date).split('-').map(Number);
  const current=Date.UTC(year,month-1,day);
  const start=year=>Date.UTC(year,0,4-new Date(Date.UTC(year,0,4)).getUTCDay());
  const first=current>=start(year+1)?start(year+1):current<start(year)?start(year-1):start(year);
  return 'S'+String(Math.floor((current-first)/604800000)+1).padStart(2,'0');
}

function forecastColor(reference,predicted){
  if(!Number.isFinite(predicted) || predicted<0 || !(reference?.pop>0) ||
     !(reference.threshold_per_100k>0))return '#94a3b8';
  const incidence=100000*predicted/reference.pop;
  return incidence>=reference.threshold_per_100k?'#fb7185':'#2dd4bf';
}

function renderCaseLegend(id,datasets,groups){
  const legend=document.getElementById(id);
  legend.replaceChildren();
  for(const group of groups){
    const items=document.createElement('ul');
    for(const [index,label] of group){
      const series=datasets[index];
      if(!series.data.some(Number.isFinite))continue;
      const item=document.createElement('li');
      const mark=document.createElement('i');
      mark.setAttribute('aria-hidden','true');
      mark.className=series.fill?'key-band':'key-line';
      if(series.fill){
        mark.style.backgroundColor=series.backgroundColor;
      }else{
        mark.style.borderTopColor=series.borderColor;
        if(series.borderDash?.length)mark.style.borderTopStyle='dashed';
      }
      item.append(mark,label);
      items.append(item);
    }
    if(items.childElementCount)legend.append(items);
  }
}
