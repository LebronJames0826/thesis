%% Initialize the data

% Clean the data
Data = rmmissing(Data);
Bt = rmmissing(DataS1);

% Generate the portfolios
P1 = Data{:,1:9};
P2 = Data{:,10:39};
P3 = Data{:,:};
Backtest = Bt{:,:};

BtP1 = [P1; Backtest(30:59, 1:9)];
BtP2 = [P2; Backtest(30:59, 10:39)];
BtP3 = [P3; Backtest(30:59, :)];

% Set the riskfree rate
riskfree_annual = 0.059250; %input the annual govt bond rate
riskfree_daily = (1+riskfree_annual)^(1/365)-1;

%% Call Generate Portfolio

[summary, p1weight, p2weight, p3weight, p1portfolioval, p2portfolioval, p3portfolioval, portfolioval] = genport(P1, P2, P3, riskfree_daily);
%% Backtest Vanilla

% [btm,btv,bts,btk,btsharpe,btcumret] = backtest(p1weight_mv, Bt, riskfree_daily);


p1bt = zeros(10,5);
p1btcumul = zeros(59,10);
for k = 1:10
    [p1bt(k,1), p1bt(k,2), p1bt(k,3), p1bt(k,4), p1bt(k,5), p1btcumul(:,k)] = backtest(p1weight(:,k), Bt, riskfree_daily, 1, 59, 1, 9);
end

l1= plot(p1btcumul);
xlabel("Day")
ylabel("Cumulative Return")
legend("Naive", "MV", "MSV", "MSV0", "MVSK1100", "MVSK1110", "MVSK1111","MVSK3111", "MVSK1310", "MVSK3311", "Location","best")
l1(8).LineStyle = "--";
l1(9).LineStyle = "--";
l1(10).LineStyle = "--";


p2bt = zeros(10,5);
p2btcumul = zeros(59,10);
for k = 1:10
    [p2bt(k,1), p2bt(k,2), p2bt(k,3), p2bt(k,4), p2bt(k,5), p2btcumul(:,k)] = backtest(p2weight(:,k), Bt, riskfree_daily, 1 ,59, 10, 39);
end

figure
l2=plot(p2btcumul);
xlabel("Day")
ylabel("Cumulative Return")
legend("Naive", "MV", "MSV", "MSV0", "MVSK1100", "MVSK1110", "MVSK1111","MVSK3111", "MVSK1310", "MVSK3311", "Location","best")
l2(8).LineStyle = "--";
l2(9).LineStyle = "--";
l2(10).LineStyle = "--";

p3bt = zeros(10,5);
p3btcumul = zeros(59,10);
for k = 1:10
    [p3bt(k,1), p3bt(k,2), p3bt(k,3), p3bt(k,4), p3bt(k,5), p3btcumul(:,k)] = backtest(p3weight(:,k), Bt, riskfree_daily, 1, 59, 1, 39);
end

figure
l3=plot(p3btcumul);
xlabel("Day")
ylabel("Cumulative Return")
legend("Naive", "MV", "MSV", "MSV0", "MVSK1100", "MVSK1110", "MVSK1111","MVSK3111", "MVSK1310", "MVSK3311", "Location","best")
l3(8).LineStyle = "--";
l3(9).LineStyle = "--";
l3(10).LineStyle = "--";
%% Backtest 1 Month Recalibration

p1btm = zeros(10,5);
p1btmcumul = zeros(30,10);
for k = 1:10
    [p1btm(k,1), p1btm(k,2), p1btm(k,3), p1btm(k,4), p1btm(k,5), p1btmcumul(:,k)] = backtest(p1weight(:,k), Bt, riskfree_daily, 30, 59, 1, 9);
end

p2btm = zeros(10,5);
p2btmcumul = zeros(30,10);
for k = 1:10
    [p2btm(k,1), p2btm(k,2), p2btm(k,3), p2btm(k,4), p2btm(k,5), p2btmcumul(:,k)] = backtest(p2weight(:,k), Bt, riskfree_daily, 30, 59, 10, 39);
end

p3btm = zeros(10,5);
p3btmcumul = zeros(30,10);
for k = 1:10
    [p3btm(k,1), p3btm(k,2), p3btm(k,3), p3btm(k,4), p3btm(k,5), p3btmcumul(:,k)] = backtest(p3weight(:,k), Bt, riskfree_daily, 30, 59, 1, 39);
end


[msummary, mp1weight, mp2weight, mp3weight, mp1portfolioval, mp2portfolioval, mp3portfolioval, mportfolioval] = genport(BtP1, BtP2, BtP3, riskfree_daily);

p1btmcumul1 = zeros(29,10);
p2btmcumul1 = zeros(29,10);
p3btmcumul1 = zeros(29,10);

for k = 1:10
    [p1btm(k,1), p1btm(k,2), p1btm(k,3), p1btm(k,4), p1btm(k,5), p1btmcumul1(:,k)] = backtest(mp1weight(:,k), Bt, riskfree_daily, 1, 29, 1, 9);
end

for k = 1:10
    [p2btm(k,1), p2btm(k,2), p2btm(k,3), p2btm(k,4), p2btm(k,5), p2btmcumul1(:,k)] = backtest(mp2weight(:,k), Bt, riskfree_daily, 1, 29, 10, 39);
end

for k = 1:10
    [p3btm(k,1), p3btm(k,2), p3btm(k,3), p3btm(k,4), p3btm(k,5), p3btmcumul1(:,k)] = backtest(mp3weight(:,k), Bt, riskfree_daily, 1, 29, 1, 39);
end

p1btmcumul_final = [p1btmcumul(:,:);p1btmcumul(30,:).*p1btmcumul1(:,:)];
p2btmcumul_final = [p2btmcumul(:,:);p2btmcumul(30,:).*p2btmcumul1(:,:)];
p3btmcumul_final = [p3btmcumul(:,:);p3btmcumul(30,:).*p3btmcumul1(:,:)];

figure
l4 = plot(p1btmcumul_final);
xlabel("Day")
ylabel("Cumulative Return")
legend("Naive", "MV", "MSV", "MSV0", "MVSK1100", "MVSK1110", "MVSK1111","MVSK3111", "MVSK1310", "MVSK3311", 'Location','best')
l4(8).LineStyle = "--";
l4(9).LineStyle = "--";
l4(10).LineStyle = "--";


figure
l5 = plot(p2btmcumul_final);
xlabel("Day")
ylabel("Cumulative Return")
legend("Naive", "MV", "MSV", "MSV0", "MVSK1100", "MVSK1110", "MVSK1111","MVSK3111", "MVSK1310", "MVSK3311", "Location","best")
l5(8).LineStyle = "--";
l5(9).LineStyle = "--";
l5(10).LineStyle = "--";

figure
l6 = plot(p3btmcumul_final);
xlabel("Day")
ylabel("Cumulative Return")
legend("Naive", "MV", "MSV", "MSV0", "MVSK1100", "MVSK1110", "MVSK1111","MVSK3111", "MVSK1310", "MVSK3311", "Location","best")
l6(8).LineStyle = "--";
l6(9).LineStyle = "--";
l6(10).LineStyle = "--";


%% Generate the portfolios

function [summary, p1weight, p2weight, p3weight, p1portfolioval, p2portfolioval, p3portfolioval, portfolioval] = genport(P1, P2, P3, riskfree_daily)

% For P1
[p1ret_mv, p1var_mv, p1skew_mv, p1kurt_mv, p1weight_mv, p1sharpe_mv, p1sortino_mv] = MeanVar(P1, 0, riskfree_daily);
[p1ret_msv, p1var_msv, p1skew_msv, p1kurt_msv, p1weight_msv, p1sharpe_msv, p1sortino_msv] = MeanSemivar(P1, 0, true, riskfree_daily);
[p1ret_msv0, p1var_msv0, p1skew_msv0, p1kurt_msv0, p1weight_msv0, p1sharpe_msv0, p1sortino_msv0] = MeanSemivar(P1, 0, 0, riskfree_daily);
[p1ret_naive, p1var_naive, p1skew_naive, p1kurt_naive, p1weight_naive, p1sharpe_naive, p1sortino_naive] = Naive(P1, riskfree_daily);
[p1ret_mvsk1111, p1var_mvsk1111, p1skew_mvsk1111, p1kurt_mvsk1111, p1weight_mvsk1111, p1sharpe_mvsk1111, p1sortino_mvsk1111, p1fval_mvsk1111] = MVSK(P1, 1, 1, 1, 1, riskfree_daily);
[p1ret_mvsk1110, p1var_mvsk1110, p1skew_mvsk1110, p1kurt_mvsk1110, p1weight_mvsk1110, p1sharpe_mvsk1110, p1sortino_mvsk1110, p1fval_mvsk1110] = MVSK(P1, 1, 1, 1, 0, riskfree_daily);
[p1ret_mvsk1100, p1var_mvsk1100, p1skew_mvsk1100, p1kurt_mvsk1100, p1weight_mvsk1100, p1sharpe_mvsk1100, p1sortino_mvsk1100, p1fval_mvsk1100] = MVSK(P1, 1, 1, 0, 0, riskfree_daily);
[p1ret_mvsk3111, p1var_mvsk3111, p1skew_mvsk3111, p1kurt_mvsk3111, p1weight_mvsk3111, p1sharpe_mvsk3111, p1sortino_mvsk3111, p1fval_mvsk3111] = MVSK(P1, 3, 1, 1, 1, riskfree_daily);
[p1ret_mvsk1311, p1var_mvsk1311, p1skew_mvsk1311, p1kurt_mvsk1311, p1weight_mvsk1311, p1sharpe_mvsk1311, p1sortino_mvsk1311, p1fval_mvsk1311] = MVSK(P1, 1, 3, 1, 1, riskfree_daily);
[p1ret_mvsk3311, p1var_mvsk3311, p1skew_mvsk3311, p1kurt_mvsk3311, p1weight_mvsk3311, p1sharpe_mvsk3311, p1sortino_mvsk3311, p1fval_mvsk3311] = MVSK(P1, 3, 3, 1, 1, riskfree_daily);

% For P2
[p2ret_mv, p2var_mv, p2skew_mv, p2kurt_mv, p2weight_mv, p2sharpe_mv, p2sortino_mv] = MeanVar(P2, 0, riskfree_daily);
[p2ret_msv, p2var_msv, p2skew_msv, p2kurt_msv, p2weight_msv, p2sharpe_msv, p2sortino_msv] = MeanSemivar(P2, 0, true, riskfree_daily);
[p2ret_msv0, p2var_msv0, p2skew_msv0, p2kurt_msv0, p2weight_msv0, p2sharpe_msv0, p2sortino_msv0] = MeanSemivar(P2, 0, 0, riskfree_daily);
[p2ret_naive, p2var_naive, p2skew_naive, p2kurt_naive, p2weight_naive, p2sharpe_naive, p2sortino_naive] = Naive(P2, riskfree_daily);
[p2ret_mvsk1111, p2var_mvsk1111, p2skew_mvsk1111, p2kurt_mvsk1111, p2weight_mvsk1111, p2sharpe_mvsk1111, p2sortino_mvsk1111, p2fval_mvsk1111] = MVSK(P2, 1, 1, 1, 1, riskfree_daily);
[p2ret_mvsk1110, p2var_mvsk1110, p2skew_mvsk1110, p2kurt_mvsk1110, p2weight_mvsk1110, p2sharpe_mvsk1110, p2sortino_mvsk1110, p2fval_mvsk1110] = MVSK(P2, 1, 1, 1, 0, riskfree_daily);
[p2ret_mvsk1100, p2var_mvsk1100, p2skew_mvsk1100, p2kurt_mvsk1100, p2weight_mvsk1100, p2sharpe_mvsk1100, p2sortino_mvsk1100, p2fval_mvsk1100] = MVSK(P2, 1, 1, 0, 0, riskfree_daily);
[p2ret_mvsk3111, p2var_mvsk3111, p2skew_mvsk3111, p2kurt_mvsk3111, p2weight_mvsk3111, p2sharpe_mvsk3111, p2sortino_mvsk3111, p2fval_mvsk3111] = MVSK(P2, 3, 1, 1, 1, riskfree_daily);
[p2ret_mvsk1311, p2var_mvsk1311, p2skew_mvsk1311, p2kurt_mvsk1311, p2weight_mvsk1311, p2sharpe_mvsk1311, p2sortino_mvsk1311, p2fval_mvsk1311] = MVSK(P2, 1, 3, 1, 1, riskfree_daily);
[p2ret_mvsk3311, p2var_mvsk3311, p2skew_mvsk3311, p2kurt_mvsk3311, p2weight_mvsk3311, p2sharpe_mvsk3311, p2sortino_mvsk3311, p2fval_mvsk3311] = MVSK(P2, 3, 3, 1, 1, riskfree_daily);

% For P3
[p3ret_mv, p3var_mv, p3skew_mv, p3kurt_mv, p3weight_mv, p3sharpe_mv, p3sortino_mv] = MeanVar(P3, 0, riskfree_daily);
[p3ret_msv, p3var_msv, p3skew_msv, p3kurt_msv, p3weight_msv, p3sharpe_msv, p3sortino_msv] = MeanSemivar(P3, 0, true, riskfree_daily);
[p3ret_msv0, p3var_msv0, p3skew_msv0, p3kurt_msv0, p3weight_msv0, p3sharpe_msv0, p3sortino_msv0] = MeanSemivar(P3, 0, 0, riskfree_daily);
[p3ret_naive, p3var_naive, p3skew_naive, p3kurt_naive, p3weight_naive, p3sharpe_naive, p3sortino_naive] = Naive(P3, riskfree_daily);
[p3ret_mvsk1111, p3var_mvsk1111, p3skew_mvsk1111, p3kurt_mvsk1111, p3weight_mvsk1111, p3sharpe_mvsk1111, p3sortino_mvsk1111, p3fval_mvsk1111] = MVSK(P3, 1, 1, 1, 1, riskfree_daily);
[p3ret_mvsk1110, p3var_mvsk1110, p3skew_mvsk1110, p3kurt_mvsk1110, p3weight_mvsk1110, p3sharpe_mvsk1110, p3sortino_mvsk1110, p3fval_mvsk1110] = MVSK(P3, 1, 1, 1, 0, riskfree_daily);
[p3ret_mvsk1100, p3var_mvsk1100, p3skew_mvsk1100, p3kurt_mvsk1100, p3weight_mvsk1100, p3sharpe_mvsk1100, p3sortino_mvsk1100, p3fval_mvsk1100] = MVSK(P3, 1, 1, 0, 0, riskfree_daily);
[p3ret_mvsk3111, p3var_mvsk3111, p3skew_mvsk3111, p3kurt_mvsk3111, p3weight_mvsk3111, p3sharpe_mvsk3111, p3sortino_mvsk3111, p3fval_mvsk3111] = MVSK(P3, 3, 1, 1, 1, riskfree_daily);
[p3ret_mvsk1311, p3var_mvsk1311, p3skew_mvsk1311, p3kurt_mvsk1311, p3weight_mvsk1311, p3sharpe_mvsk1311, p3sortino_mvsk1311, p3fval_mvsk1311] = MVSK(P3, 1, 3, 1, 1, riskfree_daily);
[p3ret_mvsk3311, p3var_mvsk3311, p3skew_mvsk3311, p3kurt_mvsk3311, p3weight_mvsk3311, p3sharpe_mvsk3311, p3sortino_mvsk3311, p3fval_mvsk3311] = MVSK(P3, 3, 3, 1, 1, riskfree_daily);


% Generate the tables
m = mean(P3)';
v = var(P3)';
skew = skewness(P3)';
kurt = kurtosis(P3)';

nrm = zeros(39,3);
for i = 1:39
[nrm(i,1), nrm(i,2), nrm(i,3)] = jbtest(P3(:,i));
end

summary = [m v skew kurt nrm];

p1weight = [p1weight_naive p1weight_mv p1weight_msv p1weight_msv0 p1weight_mvsk1100 p1weight_mvsk1110 p1weight_mvsk1111 p1weight_mvsk3111 p1weight_mvsk1311 p1weight_mvsk3311];
p1portfolioval = [0 0 0 0 p1fval_mvsk1100' p1fval_mvsk1110' p1fval_mvsk1111' p1fval_mvsk3111' p1fval_mvsk1311' p1fval_mvsk3311';
    p1ret_naive p1ret_mv p1ret_msv p1ret_msv0 p1ret_mvsk1100 p1ret_mvsk1110 p1ret_mvsk1111 p1ret_mvsk3111 p1ret_mvsk1311 p1ret_mvsk3311; 
    p1var_naive p1var_mv p1var_msv p1var_msv0 p1var_mvsk1100 p1var_mvsk1110 p1var_mvsk1111 p1var_mvsk3111 p1var_mvsk1311 p1var_mvsk3311;
    p1skew_naive p1skew_mv p1skew_msv p1skew_msv0 p1skew_mvsk1100 p1skew_mvsk1110 p1skew_mvsk1111 p1skew_mvsk3111 p1skew_mvsk1311 p1skew_mvsk3311;
    p1kurt_naive p1kurt_mv p1kurt_msv p1kurt_msv0 p1kurt_mvsk1100 p1kurt_mvsk1110 p1kurt_mvsk1111 p1kurt_mvsk3111 p1kurt_mvsk1311 p1kurt_mvsk3311;
    p1sharpe_naive p1sharpe_mv p1sharpe_msv p1sharpe_msv0 p1sharpe_mvsk1100 p1sharpe_mvsk1110 p1sharpe_mvsk1111 p1sharpe_mvsk3111 p1sharpe_mvsk1311 p1sharpe_mvsk3311;
    p1sortino_naive p1sortino_mv p1sortino_msv p1sortino_msv0 p1sortino_mvsk1100 p1sortino_mvsk1110 p1sortino_mvsk1111 p1sortino_mvsk3111 p1sortino_mvsk1311 p1sortino_mvsk3311]';

p2weight = [p2weight_naive p2weight_mv p2weight_msv p2weight_msv0 p2weight_mvsk1100 p2weight_mvsk1110 p2weight_mvsk1111 p2weight_mvsk3111 p2weight_mvsk1311 p2weight_mvsk3311];
p2portfolioval = [0 0 0 0 p2fval_mvsk1100' p2fval_mvsk1110' p2fval_mvsk1111' p2fval_mvsk3111' p2fval_mvsk1311' p2fval_mvsk3311';
    p2ret_naive p2ret_mv p2ret_msv p2ret_msv0 p2ret_mvsk1100 p2ret_mvsk1110 p2ret_mvsk1111 p2ret_mvsk3111 p2ret_mvsk1311 p2ret_mvsk3311; 
    p2var_naive p2var_mv p2var_msv p2var_msv0 p2var_mvsk1100 p2var_mvsk1110 p2var_mvsk1111 p2var_mvsk3111 p2var_mvsk1311 p2var_mvsk3311;
    p2skew_naive p2skew_mv p2skew_msv p2skew_msv0 p2skew_mvsk1100 p2skew_mvsk1110 p2skew_mvsk1111 p2skew_mvsk3111 p2skew_mvsk1311 p2skew_mvsk3311;
    p2kurt_naive p2kurt_mv p2kurt_msv p2kurt_msv0 p2kurt_mvsk1100 p2kurt_mvsk1110 p2kurt_mvsk1111 p2kurt_mvsk3111 p2kurt_mvsk1311 p2kurt_mvsk3311;
    p2sharpe_naive p2sharpe_mv p2sharpe_msv p2sharpe_msv0 p2sharpe_mvsk1100 p2sharpe_mvsk1110 p2sharpe_mvsk1111 p2sharpe_mvsk3111 p2sharpe_mvsk1311 p2sharpe_mvsk3311;
    p2sortino_naive p2sortino_mv p2sortino_msv p2sortino_msv0 p2sortino_mvsk1100 p2sortino_mvsk1110 p2sortino_mvsk1111 p2sortino_mvsk3111 p2sortino_mvsk1311 p2sortino_mvsk3311]';

p3weight = [p3weight_naive p3weight_mv p3weight_msv p3weight_msv0 p3weight_mvsk1100 p3weight_mvsk1110 p3weight_mvsk1111 p3weight_mvsk3111 p3weight_mvsk1311 p3weight_mvsk3311];
p3portfolioval = [0 0 0 0 p3fval_mvsk1100' p3fval_mvsk1110' p3fval_mvsk1111' p3fval_mvsk3111' p3fval_mvsk1311' p3fval_mvsk3311';
    p3ret_naive p3ret_mv p3ret_msv p3ret_msv0 p3ret_mvsk1100 p3ret_mvsk1110 p3ret_mvsk1111 p3ret_mvsk3111 p3ret_mvsk1311 p3ret_mvsk3311; 
    p3var_naive p3var_mv p3var_msv p3var_msv0 p3var_mvsk1100 p3var_mvsk1110 p3var_mvsk1111 p3var_mvsk3111 p3var_mvsk1311 p3var_mvsk3311;
    p3skew_naive p3skew_mv p3skew_msv p3skew_msv0 p3skew_mvsk1100 p3skew_mvsk1110 p3skew_mvsk1111 p3skew_mvsk3111 p3skew_mvsk1311 p3skew_mvsk3311;
    p3kurt_naive p3kurt_mv p3kurt_msv p3kurt_msv0 p3kurt_mvsk1100 p3kurt_mvsk1110 p3kurt_mvsk1111 p3kurt_mvsk3111 p3kurt_mvsk1311 p3kurt_mvsk3311;
    p3sharpe_naive p3sharpe_mv p3sharpe_msv p3sharpe_msv0 p3sharpe_mvsk1100 p3sharpe_mvsk1110 p3sharpe_mvsk1111 p3sharpe_mvsk3111 p3sharpe_mvsk1311 p3sharpe_mvsk3311;
    p3sortino_naive p3sortino_mv p3sortino_msv p3sortino_msv0 p3sortino_mvsk1100 p3sortino_mvsk1110 p3sortino_mvsk1111 p3sortino_mvsk3111 p3sortino_mvsk1311 p3sortino_mvsk3311]';

portfolioval = [p1portfolioval; p2portfolioval; p3portfolioval];
end


function [csmat, ckmat, return_cov] = matrices(return_matrix)
% Initial values
[rown, coln] = size(return_matrix); % get the number of assets
x0 = (1/coln)*ones(coln,1); % initial guess for portfolio weights
lb = zeros(1,coln); % weights are nonnegative (no short-selling assumption)
A = ones(1,coln); % initializing the sum constraint
b = 1;

% obtain the mean vector
return_mean = mean(return_matrix);

% obtain the covariance matrix of return_matrix
return_cov = cov(return_matrix);

% Obtain the mean-adjusted returns
ret_meanadj = return_matrix - kron(return_mean, ones(rown, 1));

% Initialize the coskewness matrix
csmat = [];

for i = 1:coln
    S = [];
    for j = 1:coln
        for k = 1:coln
            u = 0;
            for t = 1:rown
                u = u + (ret_meanadj(t,i)*ret_meanadj(t,j)*ret_meanadj(t,k));
            end
            S(j,k) = u/rown;
        end 
    end
    csmat = [csmat S];
end

%Initialize the cokurtosis matrix
ckmat = [];

for i = 1:coln
    for j = 1:coln
         K = [];
        for k = 1:coln
            for l = 1:coln
                u = 0;
                for t = 1:rown
                    u = u + (ret_meanadj(t,i)*ret_meanadj(t,j)*ret_meanadj(t,k)*ret_meanadj(t,l));
                end
                K(k,l) = u/rown;
            end
        end
        ckmat = [ckmat K];
    end
end
end
%% Mean-Variance portfolio

function [pret_mv, pvar_mv, pskew_mv, pkurt_mv, weight_mv, sharpe, sortino] = MeanVar(return_matrix, target_return, riskfree)
% Initial values
[rown, coln] = size(return_matrix); % get the number of assets
x0 = (1/coln)*ones(coln,1); % initial guess for portfolio weights
lb = zeros(1,coln); % weights are nonnegative (no short-selling assumption)
A = ones(1,coln); % initializing the sum constraint
b = 1;

% obtain the mean vector
return_mean = mean(return_matrix);

% obtain the covariance matrix of return_matrix
return_cov = cov(return_matrix);

% setting up the optimization
f_mv =@(x) x'*return_cov*x;
weight_mv = fmincon(f_mv, x0, -return_mean, -target_return, A, b, lb, []);

% portfolio returns
portfolio = return_matrix*weight_mv;

% portfolio values
pret_mv = mean(portfolio);
pvar_mv = var(portfolio);
pskew_mv = skewness(portfolio);
pkurt_mv = kurtosis(portfolio);

% Sharpe Ratio Calc
sharpe = (pret_mv - riskfree)/(pvar_mv)^(0.5);

% Sortino Ratio Calc
downside_returns = min(portfolio-riskfree, 0);
pstd_ds = sqrt(mean(downside_returns.^2));
sortino = (pret_mv - riskfree)/pstd_ds;

end

%% Mean-Semivariance Portfolio

function [pret_msv, pvar_msv, pskew_msv, pkurt_msv, weight_msv, sharpe, sortino] = MeanSemivar(return_matrix, target_return, threshold, riskfree)
% Initial values
[rown, coln] = size(return_matrix); % get the number of assets
x0 = (1/coln)*ones(coln,1); % initial guess for portfolio weights
lb = zeros(1,coln); % weights are nonnegative (no short-selling assumption)
A = ones(1,coln); % initializing the sum constraint
b = 1;

% Obtain the mean vector
return_mean = mean(return_matrix);

% Initialize the semicovariance matrix
return_semicov = zeros(coln,coln);
   
if threshold == true
    for i = 1:coln
        for j = 1:coln
            u = 0;
            for t = 1:rown
            u = u + (min([return_matrix(t,i)-return_mean(1,i), 0])*min([return_matrix(t,j)-return_mean(1,j), 0]));
            end
            return_semicov(i,j) = u/(rown);
        end
    end

else
    for i = 1:coln
        for j = 1:coln
            u = 0;
            for t = 1:rown
             u = u + (min([return_matrix(t,i)-threshold, 0])*min([return_matrix(t,j)-threshold, 0]));
            end
            return_semicov(i,j) = u/(rown);
        end
    end
end

% setting up the optimization
f_msv =@(x) x'*return_semicov*x ;
weight_msv = fmincon(f_msv, x0, -return_mean, -target_return, A, b, lb, []);

% portfolio returns
portfolio = return_matrix*weight_msv;


% portfolio values
pret_msv = mean(portfolio);
pvar_msv = var(portfolio);
pskew_msv = skewness(portfolio);
pkurt_msv = kurtosis(portfolio);

% Sharpe Ratio Calc
sharpe = (pret_msv - riskfree)/(pvar_msv)^(0.5);

% Sortino Ratio Calc
downside_returns = min(portfolio-riskfree, 0);
pstd_ds = sqrt(mean(downside_returns.^2));
sortino = (pret_msv - riskfree)/pstd_ds;

end

%% Naive (Equally-weighted) Portfolio
function [pret_naive, pvar_naive, pskew_naive, pkurt_naive, weight_naive, sharpe, sortino] = Naive(return_matrix, riskfree)
% Initial values
[rown, coln] = size(return_matrix); % get the number of assets
weight_naive = (1/coln)*ones(coln,1);

% portfolio returns
portfolio = return_matrix*weight_naive;

pret_naive = mean(portfolio);
pvar_naive = var(portfolio);
pskew_naive = skewness(portfolio);
pkurt_naive = kurtosis(portfolio);

% Sharpe Ratio Calc
sharpe = (pret_naive - riskfree)/(pvar_naive)^(0.5);

% Sortino Ratio Calc
downside_returns = min(portfolio-riskfree, 0);
pstd_ds = sqrt(mean(downside_returns.^2));
sortino = (pret_naive - riskfree)/pstd_ds;

end 

%% MVSK Portfolio
function [pret_mvsk, pvar_mvsk, pskew_mvsk, pkurt_mvsk, weight_mvsk, sharpe, sortino, fval] = MVSK(return_matrix, lambda1, lambda2, lambda3, lambda4, riskfree)
% Initial values
[rown, coln] = size(return_matrix); % get the number of assets
x0 = (1/coln)*ones(coln,1); % initial guess for portfolio weights
lb = zeros(1,coln); % weights are nonnegative (no short-selling assumption)
A = ones(1,coln); % initializing the sum constraint
b = 1;

% obtain the mean vector
return_mean = mean(return_matrix);

% obtain the covariance matrix of return_matrix
return_cov = cov(return_matrix);

% Obtain the mean-adjusted returns
ret_meanadj = return_matrix - kron(return_mean, ones(rown, 1));

% Initialize the coskewness matrix
csmat = [];

for i = 1:coln
    S = [];
    for j = 1:coln
        for k = 1:coln
            u = 0;
            for t = 1:rown
                u = u + (ret_meanadj(t,i)*ret_meanadj(t,j)*ret_meanadj(t,k));
            end
            S(j,k) = u/rown;
        end 
    end
    csmat = [csmat S];
end

%Initialize the cokurtosis matrix
ckmat = [];

for i = 1:coln
    for j = 1:coln
         K = [];
        for k = 1:coln
            for l = 1:coln
                u = 0;
                for t = 1:rown
                    u = u + (ret_meanadj(t,i)*ret_meanadj(t,j)*ret_meanadj(t,k)*ret_meanadj(t,l));
                end
                K(k,l) = u/rown;
            end
        end
        ckmat = [ckmat K];
    end
end

fret = @(x) -(return_mean*x);
fvar = @(x) x'*return_cov*x;
fskew = @(x) -(x'*csmat*(kron(x,x)))/(x'*return_cov*x)^(3/2);
fkurt = @(x) (x'*ckmat*(kron(kron(x,x),x)))/(x'*return_cov*x)^2;

options = optimoptions(@fmincon, 'MaxFunctionEvaluations', 100000, 'MaxIterations', 10000); %'MaxIterations', 1000000, 'MaxFunEvals', 1000000
weightret = fmincon(fret, x0, [], [], A, b, lb, []);
weightvar = fmincon(fvar, x0, [], [], A, b, lb, [], [], options);
weightskew = fmincon(fskew, x0, [], [], A, b, lb, [], [], options);
weightkurt = fmincon(fkurt, x0, [], [], A, b, lb, [], [], options);

pretgoal = -fret(weightret);
pvargoal = fvar(weightvar);
pskewgoal = -fskew(weightskew);
pkurtgoal = fkurt(weightkurt);

fmvsk = @(d) (1+(d(length(d)-3)/abs(pretgoal)))^lambda1 + (1+(d(length(d)-2)/abs(pvargoal)))^lambda2 + (1+(d(length(d)-1)/abs(pskewgoal)))^lambda3 + (1+(d(length(d))/abs(pkurtgoal)))^lambda4;
d0 = [ones(coln, 1)/coln; ones(4,1)/4];
dlb = [zeros(coln, 1); zeros(4,1)];
dA = [ones(coln, 1); zeros(4,1)]';
db = 1; 


nonlcon = @(d)constraints(d, return_mean, return_cov, csmat, ckmat, pretgoal, pvargoal, pskewgoal, pkurtgoal);
[all, fval] = fmincon(fmvsk, d0, [], [], dA, db, dlb, [], nonlcon, options);
weight_mvsk = all(1:length(all)-4);

% portfolio returns
portfolio = return_matrix*weight_mvsk;

pret_mvsk = mean(portfolio);
pvar_mvsk = var(portfolio);
pskew_mvsk = skewness(portfolio);
pkurt_mvsk = kurtosis(portfolio);

% Sharpe Ratio Calc
sharpe = (pret_mvsk - riskfree)/(pvar_mvsk)^(0.5);

% Sortino Ratio Calc
downside_returns = min(portfolio-riskfree, 0);
pstd_ds = sqrt(mean(downside_returns.^2));
sortino = (pret_mvsk - riskfree)/pstd_ds;
end

%% functions

function [c,ceq] = constraints(d, retmean, retcov, csmat, ckmat, prett, pvar, pskew, pkurt)
x = d(1:length(d)-4);

c = [-(retmean*x)-d(length(d)-3)+prett; (x'*retcov*x)-d(length(d)-2)-pvar; -((x'*csmat*(kron(x,x)))/(x'*retcov*x)^(3/2))-d(length(d)-1)+pskew; ((x'*ckmat*(kron(kron(x,x),x)))/(x'*retcov*x)^(2))-d(length(d))-pkurt];
ceq = [];
end

%% Backtesting

function [m,v,s,k,sharpe,cum_ret] = backtest(weight, bt_data, riskfree, a, b, c, d)

[colw, roww] = size(weight);
backtest_data = bt_data{a:b, c:d};

[rown, coln] = size(backtest_data);

portfolio = backtest_data*weight;
m = mean(portfolio);
v = var(portfolio);
s = skewness(portfolio);
k = kurtosis(portfolio);
sharpe = (m - riskfree)/v^0.5;

cum_ret = zeros(rown, 1);
adjbacktest_data = 1+backtest_data;

for i = 1:rown
u = ones(1,colw);

    for j = 1:i
    u = u.*adjbacktest_data(rown-j+1,:);
    end

cum_ret(i,1) = u*weight;
end

end

%%
[p1csmat, p1ckmat, p1covar] = matrices(P1);
[p2csmat, p2ckmat, p2covar] = matrices(P2);
[p3csmat, p3ckmat, p3covar] = matrices(P3);