function [estfinal,Fittedfinal] = expcurvefit(x,y,start_point)

minssef = inf;
options.Display = 'off';
% repeat 20 times with random start points
for i = 1:20
    
    if ~exist('start_point','var')
        start_point = [rand(1) -rand(1) rand(1)];
    end
    
    % start_point = [Aguess, Bguess, Cguess];
    est = fminsearch(@expfun, start_point,options);
    Aest = est(1);
    Best = est(2);
    Cest = est(3);
        
    Fitted = Aest + (Best.* exp(-Cest * x));
    
    ErrorVector = Fitted-y;
    ssef = sum(ErrorVector.^2);
    if ssef < minssef
        estfinal = est;
        Fittedfinal = Fitted;
        minssef = ssef;
    end
end

    function [sse, FittedCurve] = expfun(params)
        A = params(1);
        B = params(2);
        C = params(3);
        FittedCurve = A + (B .* exp(-C * x));
        ErrorVector = FittedCurve - y;
        sse = ErrorVector*ErrorVector' + 10*(ErrorVector(end)*ErrorVector(end)); % NEW!! Added extra penalty for not passing close to last point
    end

end