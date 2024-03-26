M = 2000;    % window length = embedding dimension
N = 11000;   % length of generated time series
T = 22;    % period length of sine function
stdnoise = 1; % noise-to-signal ratio

t = (1:N)';
%X = sin(2*pi*t/T);
%noise = stdnoise*randn(size(X));
%X = X + noise;
%X = X - mean(X);            % remove mean value
%X = X/std(X,1);             % normalize to standard deviation 1
[X,fs] = audioread("klattsyn.wav");
fs

figure(1);
set(gcf,'name','Time series X');
clf;
plot(t, X,'b-');

Y=zeros(N-M+1,M);
for m=1:M
  Y(:,m) = X((1:N-M+1)+m-1);
end
C=Y'*Y / (N-M+1);

[RHO,LAMBDA] = eig(C);
LAMBDA = diag(LAMBDA);               % extract the diagonal elements
[LAMBDA,ind]=sort(LAMBDA,'descend'); % sort eigenvalues
RHO = RHO(:,ind);                    % and eigenvectors

figure(2);
set(gcf,'name','Eigenvalues LAMBDA')
clf;
plot(LAMBDA(1:30),'o-');
xlabel("Singular Values", 'FontName', 'Times New Roman', 'FontSize', 12);
ylabel("Significance", 'FontName', 'Times New Roman', 'FontSize', 12);