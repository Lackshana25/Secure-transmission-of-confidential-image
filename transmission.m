clc;
clear all;
close all;
% Define parameters
n = 65535;
r1 = 3.989;
x0 = 0.449;
% Logistic map generation
x = zeros(1, n);
x(1) = x0;
for k = 1:n
    x(k+1) = r1 * x(k) * (1 - x(k));
end
a = reshape(x,[256,256]);
h = mod(a * 10^17, 256) + 1;
% Converting decimal to binary 8-bit key
for i = 1:256
    for j = 1:256
        finalOutput{i, j} = dec2bin(h(i, j), 8);
    end
end
% DNA encoding lookup table
codebook1 = containers.Map({'00','11','01','10'}, {'A','T','C','G'});
C1 = cellfun(@(x) values(codebook1, {x(1:2),x(3:4),x(5:6),x(7:8)}), ...
               finalOutput, 'uni', 0);
A = cellfun(@cell2mat, C1, 'uni', 0);
% Read input color image
input_image = imread('C:\Users\Lakshana\Downloads\4.1.02.tiff');
% Convert decimal to binary 8-bit for each channel
for channel = 1:3
    for i = 1:256
        for j = 1:256
            finalOutput1{i,j,channel} = dec2bin(input_image(i,j,channel),8);
        end
    end
end
% Convert according to DNA encoding
codebook2 = containers.Map({'00','11','01','10'},{'A','T','C','G'}); %// Lookup
for channel = 1:3
    C2 = cellfun(@(x) values(codebook2, {x(1:2),x(3:4),x(5:6),x(7:8)}), ...
                                             finalOutput1(:,:,channel), 'uni', 0);
    B{channel} = cellfun(@cell2mat, C2, 'uni', 0);
end
% DNA xor operation
for channel = 1:3
    for i=1:256
        for j=1:256
            img_a = A{i,j};
            key_b = B{channel}{i,j};
            % Perform DNA xor operation
            for bit_index = 1:4
                if (img_a(bit_index) == 'A' && key_b(bit_index) == 'A') || ...
                   (img_a(bit_index) == 'T' && key_b(bit_index) == 'T') || ...
                   (img_a(bit_index) == 'C' && key_b(bit_index) == 'C') || ...
                   (img_a(bit_index) == 'G' && key_b(bit_index) == 'G')
                    dna_a(bit_index) = 'A';
                elseif (img_a(bit_index) == 'A' && key_b(bit_index) == 'C') || ...
                   (img_a(bit_index) == 'C' && key_b(bit_index) == 'A') || ...
                   (img_a(bit_index) == 'T' && key_b(bit_index) == 'G') || ...
                   (img_a(bit_index) == 'G' && key_b(bit_index) == 'T')
                    dna_a(bit_index) = 'C';
                elseif (img_a(bit_index) == 'A' && key_b(bit_index) == 'G') || ...
                   (img_a(bit_index) == 'C' && key_b(bit_index) == 'T') || ...
                   (img_a(bit_index) == 'T' && key_b(bit_index) == 'C') || ...
                   (img_a(bit_index) == 'G' && key_b(bit_index) == 'A')
                    dna_a(bit_index) = 'G';
                else
                    dna_a(bit_index) = 'T';
                end
            end
            dna_add{i, j, channel} = dna_a;
        end
    end
end
% Convert DNA back to binary and then to decimal
codebook3= containers.Map({'A','T','C','G'},{'11','00','10','01'}); %// Lookup
for channel = 1:3
    C3 = cellfun(@(x) values(codebook3, {x(1),x(2),x(3),x(4)}), ...
         dna_add(:,:,channel), 'uni', 0);
    B11{channel}= cellfun(@cell2mat, C3, 'uni', 0);
end
% Store encrypted image data
encrypted = zeros(256, 256, 3);
for channel = 1:3
    for i=1:256
        for j=1:256
            encrypted(i,j,channel) = bin2dec(B11{channel}{i, j});
        end
    end
end
% Confusion: Shuffle the encrypted data
shuffled_indices = randperm(256*256);
shuffled_encrypted = encrypted(shuffled_indices);
% Diffusion: Spread the influence of each pixel across the entire image
diffused_encrypted = zeros(256, 256, 3);
for channel = 1:3
    for i = 1:256
        for j = 1:256
            % Define diffusion indices
            di = mod(i + 13, 256) + 1;
            dj = mod(j + 17, 256) + 1;
            % Spread the encrypted value
            diffused_encrypted(di, dj, channel) = shuffled_encrypted((i-1)*256 + j);
        end
    end
end
imwrite(uint8(diffused_encrypted), 'encrypted_image.jpg');

%% Logistic map simulation and XOR operation
scale = 10000; % determines the level of rounding
maxpoints = 20; % determines maximum values to plot
N = 50; % number of "r" values to simulate
a = 2.4; % starting value of "r"
b = 4; % final value of "r"... anything higher diverges.
rs = linspace(a,b,N); % vector of "r" values
M = 128; % number of iterations of logistics equation

% Loop through the "r" values
for j = 1:length(rs)
    r=rs(j); % get current "r"
    x=zeros(M,1); % allocate memory
    x(1) = 0.1; % initial condition (can be anything from 0 to 1)
    for i = 2:M % iterate
        x(i) = r*x(i-1)*(1-x(i-1));
    end
    % only save those unique, semi-stable values
    out{j} = unique(round(scale*x(end-maxpoints:end)));
end

% XOR operation between encrypted image and logistic map simulation results
for channel = 1:3
    for i = 1:256
        for j = 1:256
            diffused_encrypted(i,j,channel) = bitxor(uint8(diffused_encrypted(i,j,channel)), uint8(out{1}(1)));
        end
    end
end
imwrite(uint8(diffused_encrypted), 'encrypted_image.jpg');
% Reverse diffusion: Spread the influence of each pixel back to its original position
reversed_diffused_encrypted = zeros(256, 256, 3);
for channel = 1:3
    for i = 1:256
        for j = 1:256
            % Define diffusion indices
            di = mod(i - 13, 256) + 1;
            dj = mod(j - 17, 256) + 1;
            % Spread the encrypted value back to its original position
            reversed_diffused_encrypted(i, j, channel) = diffused_encrypted(di, dj, channel);
        end
    end
end

% Reverse confusion: Unshuffle the encrypted data
reversed_encrypted = zeros(256*256, 3);
for channel = 1:3
    for i = 1:256*256
        reversed_encrypted(i) = shuffled_encrypted(shuffled_indices == i);
    end
end

% Reshape the reversed encrypted data to its original form
reversed_encrypted_image = reshape(reversed_encrypted, [256, 256, 3]);

% Convert back to original image data
reversed_image = zeros(256, 256, 3);
for channel = 1:3
    for i = 1:256
        for j = 1:256
            reversed_image(i, j, channel) = bin2dec(B11{channel}{i, j});
        end
    end
end

% Convert the decrypted image data to uint8
decrypted_image = uint8(reversed_image);
for channel = 1:3
    for i = 1:256
        for j = 1:256
            decr_bin{i,j,channel} = dec2bin(reversed_image(i,j,channel), 8);
        end
    end
end 

% Convert decrypted image according to rule 7 for each channel
codebook7 = containers.Map({'11','00','10','01'},{'A','T','C','G'}); % Lookup
for channel = 1:3
    C_4 = cellfun(@(x) values(codebook7, {x(1:2),x(3:4),x(5:6),x(7:8)}), ...
                  decr_bin(:,:,channel), 'uni', 0);
    B_2{channel} = cellfun(@cell2mat, C_4, 'uni', 0);
end

% Generate Logistic map for rule 2 conversion
n = 65535;
x=zeros(1,n);
r1=3.989;
x0=0.449;
x(1)=x0;
for k=1:n
    x(k+1)=r1*x(k)*(1-x(k));
end
a = reshape(x,[256,256]);
h =[mod(a* 10^17,256)+1];

% Convert Logistic map output to binary 8-bit for each channel
for channel = 1:3
    for i = 1:256
        for j = 1:256
            finalOutput_3{i,j,channel} = dec2bin(h(i,j), 8);
        end
    end
end 

% Convert Logistic map output according to rule 2 for each channel
codebook8 = containers.Map({'00','11','01','10'},{'A','T','C','G'}); % Lookup
for channel = 1:3
    C_5 = cellfun(@(x) values(codebook8, {x(1:2),x(3:4),x(5:6),x(7:8)}), ...
                   finalOutput_3(:,:,channel), 'uni', 0);
    A_2{channel} = cellfun(@cell2mat, C_5, 'uni', 0);
end

% DNA xor operation
for channel = 1:3
    for i=1:256
        for j=1:256
            img_b = A_2{channel}{i,j};
            key_c = B_2{channel}{i,j};
            % Perform DNA xor operation
            for bit_index = 1:4
                if (img_b(bit_index) == 'A' && key_c(bit_index) == 'A') || ...
                   (img_b(bit_index) == 'T' && key_c(bit_index) == 'T') || ...
                   (img_b(bit_index) == 'C' && key_c(bit_index) == 'C') || ...
                   (img_b(bit_index) == 'G' && key_c(bit_index) == 'G')
                    dna_b(bit_index) = 'A';
                elseif (img_b(bit_index) == 'A' && key_c(bit_index) == 'C') || ...
                   (img_b(bit_index) == 'C' && key_c(bit_index) == 'A') || ...
                   (img_b(bit_index) == 'T' && key_c(bit_index) == 'G') || ...
                   (img_b(bit_index) == 'G' && key_c(bit_index) == 'T')
                    dna_b(bit_index) = 'C';
                elseif (img_b(bit_index) == 'A' && key_c(bit_index) == 'G') || ...
                   (img_b(bit_index) == 'C' && key_c(bit_index) == 'T') || ...
                   (img_b(bit_index) == 'T' && key_c(bit_index) == 'C') || ...
                   (img_b(bit_index) == 'G' && key_c(bit_index) == 'A')
                    dna_b(bit_index) = 'G';
                else
                    dna_b(bit_index) = 'T';
                end
            end
            dna_add_2{i,j,channel} = dna_b;
        end
    end
end

% Convert DNA back to binary and then to decimal using rule 2 for each channel
codebook9 = containers.Map({'A','T','C','G'},{'00','11','01','10'}); % Lookup
for channel = 1:3
    C_6 = cellfun(@(x) values(codebook9, {x(1),x(2),x(3),x(4)}), ...
                 dna_add_2(:,:,channel), 'uni', 0);
    B_12{channel} = cellfun(@cell2mat, C_6, 'uni', 0);
end

% Store decrypted RGB image data
decrypted_rgb = zeros(256, 256, 3);
for channel = 1:3
    for i=1:256
        for j=1:256
            decrypted_rgb(i,j,channel) = bin2dec(B_12{channel}{i,j});
        end
    end
end

figure;
subplot(1,3,1);
imshow(input_image);
title('Original Image');
subplot(1,3,2);
imshow(uint8(diffused_encrypted));
title('Encrypted Image');
subplot(1,3,3);
imshow(uint8(decrypted_rgb));
title('Decrypted Image');
