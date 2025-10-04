function saveFigAsInputOutput(fig_eps_x, fig_eps_y, fig_eps_xy, figMises, figTOP, index, targetSize, path_input, path_out_1, path_out_2)

    % Helper function: capture and crop figure
    function imgNorm = captureFigResized(fig, targetSize)
        if isa(fig, 'matlab.ui.Figure')
            ax = fig.CurrentAxes;
        else
            ax = fig;
        end
        frame = getframe(ax);
        img = frame.cdata;

        % Convert to grayscale
        imgGray = rgb2gray(img);

        % === Crop nonwhite region (the L-shape) ===
        mask = imgGray < 250;  % pixels below 250 are structure
        if any(mask(:))
            props = regionprops(mask, 'BoundingBox');
            bbox = props(1).BoundingBox;
            img = imcrop(img, bbox);
            imgGray = rgb2gray(img);
            mask = imgGray < 250;
        end

        % Apply mask (remove white background)
        imgGray(~mask) = 0;

        % Resize and normalize
        imgResized = imresize(imgGray, targetSize);
        imgNorm = double(imgResized) / 255;
    end

    % === Capture input channels ===
    img_eps_x  = captureFigResized(fig_eps_x, targetSize);
    img_eps_y  = captureFigResized(fig_eps_y, targetSize);
    img_eps_xy = captureFigResized(fig_eps_xy, targetSize);
    img_Mises  = captureFigResized(figMises,  targetSize);

    % === Capture output ===
    img_TOP = captureFigResized(figTOP, targetSize);

    % === Combine into input arrays ===
    X = cat(3, img_eps_x, img_eps_y, img_eps_xy);  % H × W × 3
    X_Mises = img_Mises;                           % H × W
    Y = img_TOP;                                   % H × W

    % === Ensure directories exist ===
    if ~exist(path_input, 'dir')
        mkdir(path_input);
    end
    if ~exist(path_out_1, 'dir')
        mkdir(path_out_1);
    end
    if ~exist(path_out_2, 'dir')
        mkdir(path_out_2);
    end

    % === Save ===
    save(fullfile(path_input, ['input_', num2str(index), '.mat']), 'X');
    save(fullfile(path_out_1, ['output_', num2str(index), '.mat']), 'Y');
    save(fullfile(path_out_2, ['output_Mises_', num2str(index), '.mat']), 'X_Mises');

end
