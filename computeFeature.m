function database = computeFeature(database, net, useGpu, useFlip)

for ii = 1:length(database.cname)
    mkdir([database.fea_dir '/' database.cname{ii}]);
    if useFlip
        mkdir([database.fea_flip_dir '/' database.cname{ii}]);
    end
end


% compute the features for each images
for ii = 1:length(database.path)
    imgpath = database.path{ii};
    fprintf('%4d:%s\n',ii,imgpath);
    im = imread(imgpath) ;
    
    % obtain and preprocess an image
    im_ = single(im) ; % note: 255 range
    
    % deal with gray-scale image (Just in case)
    if size(im_,3) == 1
        im_(:,:,2) = im_(:,:,1);
        im_(:,:,3) = im_(:,:,1);
    end
    
    %
    im_ = imresize(im_, net.normalization.imageSize(1:2)) ;
    im_ = im_ - net.normalization.averageImage ;
    
    % run the CNN
    if useGpu
        im_ = gpuArray(im_);
    end
    res = vl_simplenn(net, im_) ;
    feature = squeeze(gather(res(end-2).x));  % Get the 4096 dims features
    [path, name] = fileparts(imgpath);
    path = strrep(path,'\','/');
    feapath = strrep(path, database.img_dir, database.fea_dir);
    save([feapath '/' name '.mat'],'feature');
    database.fea_path{ii} = [feapath '/' name '.mat'];
    
    % left-right flip image
    if useFlip
        im_flip = flip(im_,2);  
        if useGpu
            im_flip = gpuArray(im_flip);
        end
        res_flip = vl_simplenn(net, im_flip);
        feature = squeeze(gather(res_flip(end-2).x)); % Get the 4096 dims features
        [path, name] = fileparts(imgpath);
        path = strrep(path,'\','/');
        feapath = strrep(path, database.img_dir, database.fea_flip_dir);
        save([feapath '/' name '.mat'],'feature');
        database.fea_flip_path{ii} = [feapath '/' name '.mat'];
    end
end
