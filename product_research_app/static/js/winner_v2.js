(function(global){
  const metricDefs = [
    {key:'magnitud_deseo',label:'Magnitud deseo',tip:'Nivel de deseo del producto'},
    {key:'nivel_consciencia_headroom',label:'Headroom de consciencia',tip:'Conocimiento del problema/solución'},
    {key:'evidencia_demanda',label:'Evidencia demanda',tip:'Demanda observada'},
    {key:'tasa_conversion',label:'Tasa conversión',tip:'Porcentaje de conversión'},
    {key:'ventas_por_dia',label:'Ventas por día',tip:'Unidades vendidas por día'},
    {key:'recencia_lanzamiento',label:'Recencia lanzamiento',tip:'Tiempo desde lanzamiento'},
    {key:'competition_level_invertido',label:'Competencia (invertido)',tip:'Nivel de competencia inverso'},
    {key:'facilidad_anuncio',label:'Facilidad anuncio',tip:'Facilidad para anunciar'},
    {key:'escalabilidad',label:'Escalabilidad',tip:'Capacidad de escalar'},
    {key:'durabilidad_recurrencia',label:'Durabilidad/recurrencia',tip:'Durabilidad o recurrencia'}
  ];

  const MAPS = {
    magnitud_deseo:{low:0.33, medium:0.66, high:1.0},
    nivel_consciencia_headroom:{unaware:1, problem:0.8, solution:0.6, product:0.4, most:0.2},
    competition_level_invertido:{low:1.0, medium:0.5, high:0.0},
    facilidad_anuncio:{low:0.33, med:0.66, medium:0.66, high:1.0},
    escalabilidad:{low:0.33, med:0.66, medium:0.66, high:1.0},
    durabilidad_recurrencia:{consumible:1.0, durable:0.0, intermedio:0.5}
  };

  function getDefaultWeightsV2(){
    return {
      lanzamiento:{
        magnitud_deseo:70,
        nivel_consciencia_headroom:60,
        evidencia_demanda:50,
        tasa_conversion:40,
        ventas_por_dia:50,
        recencia_lanzamiento:70,
        competition_level_invertido:60,
        facilidad_anuncio:60,
        escalabilidad:40,
        durabilidad_recurrencia:40
      },
      rentabilidad:{
        evidencia_demanda:80,
        ventas_por_dia:70,
        tasa_conversion:60,
        durabilidad_recurrencia:60,
        competition_level_invertido:50,
        magnitud_deseo:50,
        nivel_consciencia_headroom:40,
        facilidad_anuncio:40,
        escalabilidad:50,
        recencia_lanzamiento:30
      }
    };
  }

  function clamp(v){
    if(v<0) return 0; if(v>1) return 1; return v;
  }

  function normalizeMetric(name, value, ranges={}){
    if(value==null) return null;
    switch(name){
      case 'magnitud_deseo':
      case 'nivel_consciencia_headroom':
      case 'competition_level_invertido':
      case 'facilidad_anuncio':
      case 'escalabilidad':
      case 'durabilidad_recurrencia':
        return MAPS[name][String(value).toLowerCase()] ?? null;
      case 'evidencia_demanda':{
        const v = Math.log1p(Number(value)||0);
        const r = ranges[name]||{}; const min=r.p5??0; const max=r.p95??1; return clamp((v-min)/(max-min||1));
      }
      case 'tasa_conversion':
        return clamp((Number(value)||0)/100);
      case 'ventas_por_dia':{
        const v = Number(value)||0; const r=ranges[name]||{}; const min=r.p5??0; const max=r.p95??1; return clamp((v-min)/(max-min||1));
      }
      case 'recencia_lanzamiento':
        return Math.exp(-((Number(value)||0)/180));
      default:
        return null;
    }
  }

  function computeRanges(list){
    const nums={evidencia_demanda:[], ventas_por_dia:[]};
    list.forEach(p=>{
      if(p.evidencia_demanda!=null) nums.evidencia_demanda.push(Math.log1p(Number(p.evidencia_demanda)||0));
      if(p.ventas_por_dia!=null) nums.ventas_por_dia.push(Number(p.ventas_por_dia)||0);
    });
    const ranges={};
    for(const k in nums){
      const arr=nums[k].sort((a,b)=>a-b);
      if(arr.length){
        const p5=arr[Math.floor(arr.length*0.05)];
        const p95=arr[Math.floor(arr.length*0.95)];
        ranges[k]={p5,p95};
      }
    }
    return ranges;
  }

  function scoreProduct(prod, weights, ranges){
    let totalW=0; let score=0;
    for(const k in weights){
      const w=weights[k];
      const norm=normalizeMetric(k, prod[k], ranges);
      if(norm!=null){
        totalW+=w;
        score+=w*norm;
      }
    }
    if(totalW<=0) return 0;
    return score/totalW;
  }

  global.winnerV2 = {metricDefs, MAPS, normalizeMetric, computeRanges, scoreProduct, getDefaultWeightsV2};
})(window);
