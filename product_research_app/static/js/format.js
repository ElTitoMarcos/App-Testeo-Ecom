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

