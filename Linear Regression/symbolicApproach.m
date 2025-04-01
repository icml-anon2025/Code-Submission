

% Load the dataset from the workspace
df = cleanedairquality;

% Limit to the first 500 rows
nRows = min(500, height(df)); % Ensure it does not exceed available rows
df = df(1:nRows, :);

% Display the DataFrame
disp('DataFrame:');
disp(head(df, 10));

disp('Replacing missing values with unique symbols...');
% Convert the table to a cell array to handle symbolic replacements
df_cell = table2cell(df);

% Step 1: Replace missing values with unique symbols
for col = 1:width(df)
    if any(ismissing(df{:, col}))
        symbol = sym(sprintf('missing_%s', df.Properties.VariableNames{col}));
        for row = 1:height(df)
            if ismissing(df{row, col})
                df_cell{row, col} = symbol;
            else
                df_cell{row, col} = sym(df_cell{row, col}); % Convert numerical values to symbolic
            end
        end
    else
        % Convert entire column to symbolic if no missing values
        df_cell(:, col) = num2cell(sym(cell2mat(df_cell(:, col))));
    end
end

% Convert the modified cell array back to a table
df = cell2table(df_cell, 'VariableNames', df.Properties.VariableNames);

disp('Modified DataFrame:');
disp(head(df, 10));

% Extract label vector and feature matrix
label_vector = sym(df{:, strcmp(df.Properties.VariableNames, 'T')});
feature_vectors = sym(df{:, ~strcmp(df.Properties.VariableNames, 'T')});

% Step 3: Calculate the residual vector
disp('Calculating residual...');
residual_vector = calculate_residual(label_vector, feature_vectors);
disp('Residual vector:');
disp(residual_vector);

% Step 4: Define vector b
b = label_vector - residual_vector;

% Step 5: Solve the system A * x = b
A = feature_vectors;
disp('Solving the system A * x = b...');

try
    % Attempt to solve using LU decomposition
    solution = linsolve(A, b);
catch
    disp('Matrix is non-invertible, using pinv instead...');
    solution = pinv(A) * b;
end

% Limit precision of the solution
solution = vpa(solution, 5);

% Step 7: Pretty print the solution
disp('Formatting solution...');
feature_names = df.Properties.VariableNames(~strcmp(df.Properties.VariableNames, 'T'));
solution_dict = containers.Map(feature_names, num2cell(solution));

% Step 8: Convert the solution to a table for better visualization
solution_table = cell2table([keys(solution_dict); values(solution_dict)]');
solution_table.Properties.VariableNames = {'Feature', 'Coefficient'};
disp('Coefficients Table:');
disp(solution_table);

% Define the project function
function proj = project(a, b)
    proj = (dot(a, b) / dot(b, b)) * b;
end

% Define the calculate_residual function
function residual = calculate_residual(label_vector, feature_vectors)
    residual = label_vector;
    for col = 1:size(feature_vectors, 2)
        feature = feature_vectors(:, col);
        residual = residual - project(residual, feature);
    end
end