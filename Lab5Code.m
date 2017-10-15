%% Load Data
% Create Image Set Vector
imgSetVector = imageSet('C:\Users\samya\Documents\MATLAB\Machine Learning\Machine Learning Lab 4\orl_faces\Train', 'recursive');
imgSetVector_test = imageSet('C:\Users\samya\Documents\MATLAB\Machine Learning\Machine Learning Lab 4\orl_faces\Test', 'recursive');

%% Main Code
% Reshape and read into a large vector for training data
k = 1;
for i = 1:length(imgSetVector)
    for j = 1:9
    face(:,k) = reshape(read(imgSetVector(i), j), [112*92, 1]);
    k = k + 1;
    end
end

% Reshape and read into a large vector for test data. We can then use this
% for the test data for facial recognition. 
k = 1;
for i = 1:length(imgSetVector_test)
    face_test(:,k) = reshape(read(imgSetVector_test(i), 1), [112*92, 1]);
    k = k + 1;
end

% Show original image of one of the faces
figure;
imshow(reshape(face(:,3), [112, 92]));

% Calculate mean
meanmat = mean(face, 2);

% Centralize all data
face_cen = double(face) - meanmat;

% Create correlation matrix A'A for better efficiency
Corr_mat = face_cen' * face_cen;

% Calculate Eigenvectors and Eigenvalues
[EigVec, EigVal] = eig(Corr_mat);

% Take diagonal of all eigenvalues and flip them so we now have largest
% first. Do the same with vectors.
EigVal = diag(EigVal);
EigVec = fliplr(EigVec);
EigVal = fliplr(EigVal');

% Bring back into eigenface space
eigFace = face_cen * EigVec;

%Plot the Eigenvalues
figure;
plot(EigVal);
title('Eigenvalues plot');
ylabel('Eigenvalue');
xlabel('Index');

% Normalize the data by dividing by the square root of the eigenvalues
for i = 1:360
V_norm(:,i) = eigFace(:,i) / sqrt(EigVal(i));
end

%%%%%%%%%%%%%%%

% Plot the eigenfaces
figure
subplot(2,2,1);
eigFace1 = reshape(V_norm(:,1), [112, 92]);
imagesc(eigFace1);
title(strcat('EigenFace 1', num2str(i)));

subplot(2,2,2);
eigFace2 = reshape(V_norm(:,2), [112, 92]);
imagesc(eigFace2);
title('EigenFace 2');

subplot(2,2,3);
eigFace3 = reshape(V_norm(:,3), [112, 92]);
imagesc(eigFace3);
title('EigenFace 3');

subplot(2,2,4);
eigFaceEnd = reshape(V_norm(:,360), [112, 92]);
imagesc(eigFaceEnd);
title('EigenFace 360');

%%%%%%%%%%%%%%%%%%%%%%

Recon1 = dot(face_cen(:,3), V_norm(:,1))*V_norm(:,1);
Recon2 = dot(face_cen(:,3), V_norm(:,1))*V_norm(:,1) + dot(face_cen(:,3), V_norm(:,2))*V_norm(:,2);
Recon3 = dot(face_cen(:,3), V_norm(:,1))*V_norm(:,1) + dot(face_cen(:,3), V_norm(:,2))*V_norm(:,2) + dot(face_cen(:,3), V_norm(:,3))*V_norm(:,3) + dot(face_cen(:,3), V_norm(:,4))*V_norm(:,4);

figure;
subplot(3,1,1);
imshow(reshape(Recon1 + meanmat, [112, 92]),[]);
title('1 Eigenfaces');

subplot(3,1,2);
imshow(reshape(Recon2 + meanmat, [112, 92]),[]);
title('2 Eigenfaces');

subplot(3,1,3);
imshow(reshape(Recon3 + meanmat, [112, 92]),[]);
title('4 Eigenfaces');

%% Calculating all weights

for j = 1:360
    Recon1 = zeros(40, 1);
    for i=1:40
        Recon1(i,1) = dot(face_cen(:,j), V_norm(:,i));
    end
    Weights(:,j) = Recon1;
end

%% Projecting New Faces 

testface_cen = double(face_test) - meanmat;

% Project the new faces onto the subspace and obtain the weights

for j = 1:40
    Recon1 = zeros(40, 1);
    for i=1:40
        Recon1(i,1) = dot(testface_cen(:,j), V_norm(:,i));
    end
    Weights_test(:,j) = Recon1;
end

% Find the Euclidean distance between the test face weights and the
% training weights.
e = [];
for j = 1:40
    for i = 1:360
        DiffWeight(:,i) = Weights_test(:,j) - Weights(:,i);
    end
    
    for i = 1:360
        mag = norm(DiffWeight(:,i));
        e = [e mag];
    end
    
    MinVal = min(e);
    Val(j) = find(e == MinVal); 
    e = [];
end

% Set the edges to discretize and find the person number which matches the
% input face.
edges = 1:9:361;

% New array using the classified faces
ValClass = discretize(Val,edges,'IncludedEdge','right');

%% Plotting the first 4 test images, and corresponding closest images
figure;
for j = 1:4
Recon10 = zeros(10304, 1);
for i=1:40
    Recon10 = Recon10 + dot(face_cen(:,Val(j)), V_norm(:,i))*V_norm(:,i);
end
    subplot(4,4,2 + (j-1)*(8/2));
    imshow(reshape(Recon10 + meanmat, [112,92]),[]);
    if j == 1
        title('Least Error Proj Training');
    end
end

for j = 1:4
Recon10 = zeros(10304, 1);
for i=1:40
    Recon10 = Recon10 + Weights_test(i,j)*V_norm(:,i);
end
    subplot(4,4,1 + (j-1)*(8/2));
    imshow(reshape(Recon10 + meanmat, [112,92]),[]);
    if j == 1
        title('Proj Test Image');
    end
end

for j = 1:4
    subplot(4,4,4 + (j-1)*(8/2));
    imshow(reshape(face_cen(:,Val(j)) + meanmat, [112,92]),[]);
    if j == 1
        title('Original Training Image');
    end
end

for j = 1:4
    subplot(4,4,3 + (j-1)*(8/2));
    imshow(reshape(testface_cen(:,j) + meanmat, [112,92]),[]);
    if j == 1
        title('Original Test Image');
    end
end

 %%
% % Calculate the weightings using dot product, and multiply that by the
% % eigenfaces for 10,20,30 and 40 eigenfaces.
% Recon10 = zeros(10304, 1);
% for i=1:10
%     Recon10 = Recon10 + dot(face_cen(:,3), V_norm(:,i))*V_norm(:,i);
% end
% 
% % Add the mean back
% Recon10 = Recon10+meanmat;
% 
% % Plot the figure
% figure;
% subplot(4,3,1);
% imshow(reshape(Recon10, [112, 92]),[]);
% title('10 Eigenfaces');
% 
% Recon20 = zeros(10304, 1);
% for i=1:20
%     Recon20 = Recon20 + dot(face_cen(:,3), V_norm(:,i))*V_norm(:,i);
% end
% 
% Recon20 = Recon20+meanmat;
% 
% subplot(4,3,4);
% imshow(reshape(Recon20, [112, 92]),[]);
% title('20 Eigenfaces');
% 
% Recon30 = zeros(10304, 1);
% for i=1:30
%     Recon30 = Recon30 + dot(face_cen(:,3), V_norm(:,i))*V_norm(:,i);
% end
% 
% Recon30 = Recon30+meanmat;
% 
% subplot(4,3,7);
% imshow(reshape(Recon30, [112, 92]),[]);
% title('30 Eigenfaces');
% 
% Recon40 = zeros(10304, 1);
% for i=1:40
%     Recon40 = Recon40 + dot(face_cen(:,3), V_norm(:,i))*V_norm(:,i);
% end
% 
% Recon40 = Recon40+meanmat;
% 
% subplot(4,3,10);
% imshow(reshape(Recon40, [112, 92]),[]);
% title('40 Eigenfaces');
% 
% % Plot the difference figures by subtrating the faces in eigenspace from
% % the original faces.
% subplot(4,3,3);
% imshow(reshape(double(face(:,3)) - Recon10,[112,92]),[]);
% title('Differences');
% 
% subplot(4,3,6);
% imshow(reshape(double(face(:,3)) - Recon20,[112,92]),[]);
% 
% subplot(4,3,9);
% imshow(reshape(double(face(:,3)) - Recon30,[112,92]),[]);
% 
% subplot(4,3,12);
% imshow(reshape(double(face(:,3)) - Recon40,[112,92]),[]);
% 
% % Plot the original images.
% subplot(4,3,2);
% imshow(reshape(face(:,3),[112,92]),[]);
% title('Original Images')
% 
% subplot(4,3,5);
% imshow(reshape(face(:,3),[112,92]),[]);
% 
% subplot(4,3,8);
% imshow(reshape(face(:,3),[112,92]),[]);
% 
% subplot(4,3,11);
% imshow(reshape(face(:,3),[112,92]),[]);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%
% 

