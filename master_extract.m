clear; clc; close all;

% Runs extraction scripts in sequence
disp('Running extract_DHT...');
run('extract_DHT.m');
disp('extract_DHT Done');

disp('Running extract_RDT...');
run('extract_RDT.m');
disp('extract_RDT Done');

disp('Running extract_ADT...');
run('extract_ADT.m');
disp('extract_ADT Done');