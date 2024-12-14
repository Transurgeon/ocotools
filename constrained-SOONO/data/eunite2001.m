min = 464; % smallest max-load in 1997-1998
max = 876; % largest max-load in 1997-1998

[y,x] = libsvmread('eunite2001');
m = svmtrain(y, x, '-s 3 -c 4096 -g 0.0625 -p 0.5');

[ty, tx] = libsvmread('eunite2001.t');

p = zeros(31,1)
for i=1:31,
  if i==1,
    txi = tx(i,:);
  else
    txi = [tx(i,1:9) (p(i-1)-min)/(max-min) tx(i-1,10:15)];
  end
  p(i) = svmpredict(ty(i), txi, m);
end

mape = 100/31*sum(abs((p-ty)./ty))
mse = (p-ty)'*(p-ty)/31
plot((1:31)', p, '--', (1:31)', ty, '-');
legend('predicted', 'real');
set(gca, 'fontsize', 18) ; 
set(findobj('Type', 'line'), 'LineWidth', 3)  
