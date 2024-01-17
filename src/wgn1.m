function y = wgn(varargin)
%WGN Generate white Gaussian noise.
%   Y = WGN(M,N,P) generates an M-by-N matrix of white Gaussian noise. P
%   specifies the power of the output noise in dBW. The unit of measure for
%   the output of the wgn function is Volts. For power calculations, it is
%   assumed that there is a load of 1 Ohm.
%
%   Y = WGN(M,N,P,IMP) specifies the load impedance in Ohms.
%
%   Y = WGN(M,N,P,IMP,S) uses S to generate random noise samples with the
%   RANDN function. S can be a random number stream specified by
%   RandStream. S can also be an integer, which seeds a random number
%   stream inside the WGN function. If you want to generate repeatable
%   noise samples, then either reset the random stream input before calling
%   WGN or use the same seed input.
%
%   Additional flags that can follow the numeric arguments are:
%
%   Y = WGN(..., POWERTYPE) specifies the units of P.  POWERTYPE can be
%   'dBW', 'dBm' or 'linear'.  Linear power is in Watts.
%
%   Y = WGN(..., OUTPUTTYPE  ); Specifies the output type.  OUTPUTTYPE can be
%   'real' or 'complex'.  If the output type is complex, then P is divided
%   equally between the real and imaginary components.
%
%   Example 1:
%       % To generate a 1024-by-1 vector of complex noise with power
%       % of 5 dBm across a 50 Ohm load, use:
%       Y = wgn(1024, 1, 5, 50, 'dBm', 'complex')
%
%   Example 2:
%       % To generate a 256-by-5 matrix of real noise with power
%       % of 10 dBW across a 1 Ohm load, use:
%       Y = wgn(256, 5, 10, 'real')
%
%   Example 3:
%       % To generate a 1-by-10 vector of complex noise with power
%       % of 3 Watts across a 75 Ohm load, use:
%       Y = wgn(1, 10, 3, 75, 'linear', 'complex')
%
%   See also RANDN, AWGN.

%   Copyright 1996-2018 The MathWorks, Inc.

%#codegen

% --- Initial checks
narginchk(3,7);

% --- Value set indicators (used for the strings)
pModeSet    = 0;
cplxModeSet = 0;

% --- Set default values
pMode    = 'dbw';
imp      = 1;
cplxMode = 'real';
seed     = 0;

% --- Placeholders for the numeric and string index values
numArg = zeros(1, nargin);
strArg = zeros(1, nargin);

numArgCounter = 0;
strArgCounter = 0;

% --- Identify string and numeric arguments
%     An empty in position 4 (Impedance) or 5 (Seed) are considered numeric
isStreamHandle = false;
isSeedSet = false;
for n=1:nargin
    if(isempty(varargin{n}))
        switch n
            case 4
                coder.internal.errorIf(comm.internal.utilities.isCharOrStringScalar(varargin{n}), 'comm:wgn:InvalidDefaultImp');
                varargin{n} = imp; % Impedance has a default value
            case 5
                coder.internal.errorIf(comm.internal.utilities.isCharOrStringScalar(varargin{n}), 'comm:wgn:InvalidNumericInput');
                varargin{n} = [];  % Seed has no default
            otherwise
                varargin{n} = '';
        end
    end
    
    % --- Assign the string and numeric vectors
    if(comm.internal.utilities.isCharOrStringScalar(varargin{n})) % If n-th argument is 'char'
        strArg(strArgCounter+1) = n;
        strArgCounter = strArgCounter + 1;
    elseif(isnumeric(varargin{n}))
        numArg(numArgCounter+1) = n;
        numArgCounter = numArgCounter + 1;
    elseif(isa(varargin{n},'RandStream'))
        numArg(numArgCounter+1) = n;
        numArgCounter = numArgCounter + 1;
        isStreamHandle = true;
    else
        coder.internal.error('comm:wgn:InvalidArg');
    end
end

% --- Build the numeric argument set
switch(nnz(numArg)) % Count of non-zero elements
    
    case 3
        % --- row is first (element 1), col (element 2), p (element 3)
        coder.internal.errorIf(~(all(numArg(1:3) == [1 2 3])), 'comm:wgn:InvalidSyntax');
        row    = varargin{numArg(1)};
        col    = varargin{numArg(2)};
        p      = varargin{numArg(3)};
        
    case 4
        % --- row is first (element 1), col (element 2), p (element 3), imp (element 4)
        coder.internal.errorIf(~(all(numArg(1:3) == [1 2 3])), 'comm:wgn:InvalidSyntax');
        row    = varargin{numArg(1)};
        col    = varargin{numArg(2)};
        p      = varargin{numArg(3)};
        imp    = varargin{numArg(4)};
        
    case 5
        % --- row is first (element 1), col (element 2), p (element 3), imp (element 4), seed (element 5)
        coder.internal.errorIf(~(all(numArg(1:3) == [1 2 3])), 'comm:wgn:InvalidSyntax');
        row    = varargin{numArg(1)};
        col    = varargin{numArg(2)};
        p      = varargin{numArg(3)};
        imp    = varargin{numArg(4)};
        seed   = varargin{numArg(5)};
        isSeedSet = true;
        
    otherwise
        coder.internal.error('comm:wgn:InvalidSyntax');
end

% --- Build the string argument set
for n=1:nnz(strArg)
    switch lower(varargin{strArg(n)})
        case {'dbw' 'dbm' 'linear'}
            if(~pModeSet)
                pModeSet = 1;
                pMode = lower(varargin{strArg(n)});
            else
                coder.internal.error('comm:wgn:TooManyPowerTypes');
            end
        case {'db'}
            coder.internal.error('comm:wgn:InvalidPowerType');
        case {'real' 'complex'}
            if(~cplxModeSet)
                cplxModeSet = 1;
                cplxMode = lower(varargin{strArg(n)});
            else
                coder.internal.error('comm:wgn:TooManyOutputTypes');
            end
        otherwise
            coder.internal.error('comm:wgn:InvalidArgOption');
    end
end

% --- Arguments and defaults have all been set, either to their defaults or by the values passed in
%     so, perform range and type checks

% --- p
if(isempty(p))
    coder.internal.error('comm:wgn:InvalidPowerVal');
end

if(any([~isreal(p) (length(p)>1) (isempty(p))]))
    coder.internal.error('comm:wgn:InvalidPowerVal');
end

if(strcmp(pMode,'linear'))
    coder.internal.errorIf(p < 0, 'comm:wgn:NegativePower');
end

% --- Dimensions
coder.internal.errorIf((any([isempty(row) isempty(col) ~isscalar(row) ~isscalar(col)])), ...
    'comm:wgn:InvalidDims');

coder.internal.errorIf((any([(row<=0) (col<=0) ~isreal(row) ~isreal(col) ((row-floor(row))~=0) ((col-floor(col))~=0)])), ...
    'comm:wgn:InvalidDims');

% --- Impedance
coder.internal.errorIf((any([~isreal(imp) (length(imp)>1) (isempty(imp)) any(imp<=0)])), ...
    'comm:wgn:InvalidImp');

% --- Seed
if(isSeedSet)
    if ~isStreamHandle
        validateattributes(seed, {'double', 'RandStream'}, ...
            {'real', 'scalar', 'integer'}, 'WGN', 'S');
    end
end

% --- All parameters are valid, so no extra checking is required
switch lower(pMode)
    case 'linear'
        noisePower = p;
    case 'dbw'
        noisePower = 10^(p/10);
    otherwise % 'dbm'
        noisePower = 10^((p-30)/10);
end

% --- Generate the noise
if isSeedSet
    if isa(seed, 'RandStream')
        stream = seed;
    elseif isempty(coder.target)
        stream = RandStream('shr3cong', 'Seed', seed);
    else
        stream = coder.internal.RandStream('shr3cong', 'Seed', seed);
    end
    
    if strcmp(cplxMode, 'complex')
        y = sqrt(imp*noisePower/2)* (randn(stream, row, col) + ...
            1i*randn(stream, row, col));
    else
        y = sqrt(imp*noisePower)* randn(stream, row, col);
    end
else
    if strcmp(cplxMode, 'complex')
        y = sqrt(imp*noisePower/2)* (randn(row, col) + 1i*randn(row, col));
    else
        y = sqrt(imp*noisePower)* randn(row, col);
    end
end
