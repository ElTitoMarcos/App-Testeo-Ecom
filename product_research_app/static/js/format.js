export function abbr(n){
  if(n>=1e6) return (n/1e6).toFixed(2)+' M';
  if(n>=1e3) return (n/1e3).toFixed(2)+' K';
  return String(n);
}

export function winnerScoreClass(s){
  if(s>=80) return 'badge score-green';
  if(s>=60) return 'badge score-amber';
  return 'badge score-red';
}

export function fmtNumber(n, dec = 0) {
  return Number(n || 0).toLocaleString('es-ES', {
    minimumFractionDigits: dec,
    maximumFractionDigits: dec,
  });
}

export function fmtInt(n) {
  return fmtNumber(n, 0);
}

export function fmtPrice(n) {
  return fmtNumber(n, 2);
}

export function fmtFloat2(n) {
  return fmtNumber(n, 2);
}

export function fmtPct(n) {
  return fmtNumber(n, 1) + '%';
}

export function nameWithFlames(name/*, trendingScore*/) {
  return String(name ?? "");
}

