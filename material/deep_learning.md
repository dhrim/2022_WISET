
# 딥러닝
    - 딥러닝 기초 - DNN, CNN, RNN 등
    - 딥러닝을 이용한 영상 데이터 분석
    - 시계열 금융데이터 분석
    - 딥러닝을 이용한 추천 서비스 분석

<br>

# 실습 자료들

[practice](../material/deep_learning/practice)


<br>

# 실습 데이터들

- 속성 데이터
    - 신용카드 연체 예측 : https://dacon.io/competitions/official/235713/data, 카드 사용자의 정보를 가지고 credit을 분류, 단순 속성 분류 문제
    - 주차 수요 예측 : https://dacon.io/competitions/official/235745/data, 아파트 단지의 속성 데이터로 주차수 예측, 단순 속성 예측 문제
    - 구내 식당 식수 예측 : https://dacon.io/competitions/official/235743/data, 사내 속성 데이터로 구내 식수 예측, 단순 속성 예측 문제 
- 영상 데이터
    - 생육 기간 예측 : https://dacon.io/competitions/official/235851/data, 시간에 따른 작물의 사진으로 학습.2개의 사진에 대해 경과한 일자를 예측, 2개의 영상을 입력으로 경과 시간 값 예측 문제
    - 병변 검출 : https://dacon.io/competitions/official/235855/data, 이미지에서 특정 대상 탐지., object detection 문제
    - 작물 병해 분류 : https://dacon.io/competitions/official/235842/data, 작물 영상으로 병해 분류, 단순 영상 분류 문제
- 순차열 데이터
    - 주가 예측 : https://dacon.io/competitions/official/235857/data, 순차 데이터 주가 데이터를 사용하여 다음날 주가 예측, 단순 순차열 예측 문제
    - 영어 음성 국적 분류 : https://dacon.io/competitions/official/235738/data 음성 순차열에 대한 국적 분류, 단순 순차열 분류 문제
- 복합 데이터  
    - 작물 병해 진단 : https://dacon.io/competitions/official/235870/data, 영상과 환경 속성 순차열 데이터를 가지고 작물 종류, 작물 상태, 질병 피해 정도를 예측, 영상과 환경의 다른 타입 입력으로 3가지 종류의 분류 문제


# 인공지능의 이해
- 인공지능 개념 및 동작 원리의 이해 : [deep_learning_intro.pptx](../material/deep_learning/deep_learning_intro.pptx)
    - Perceptron, MLP, DNN 소개
    - DNN의 학습 이해
    - AI, 머신러닝, 딥러닝의 이해
    - 딥러닝 상세 기술 이해

<br>

# 개발 환경

- 딥러닝 개발 환경
- 기본 linux 명령의 이해와 실습 : [linux.md](../material/linux.md)
- jupyter와 colab 이해 : [jupyter_and_colab.md](../material/env/jupyter_and_colab.md)


<br>

# Keras

- Keras 파악, 딥러닝 코드 파악 실습 : : [dnn_in_keras.ipynb](../material/deep_learning/dnn_in_keras.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/dnn_in_keras.ipynb)
    - 표준 Keras 딥러닝 코드
    - 로스 보기
    - 은닉층과 노드 수
    - trian, test 데이터 분리
    - batch size와 학습
    - 데이터 수와 학습
    - normalization
    - 모델 저장과 로딩
    - 노이즈 내구성
    - GPU 설정
    - 데이터 수와 성능
    - 다양한 입출력
    - callback : [dnn_in_keras_callback.ipynb](../material/deep_learning/dnn_in_keras_callback.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/dnn_in_keras_callback.ipynb)
    - overfitting 처리 : [dnn_in_keras_overfitting.ipynb](../material/deep_learning/dnn_in_keras_overfitting.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/dnn_in_keras_overfitting.ipynb)
    - custom data generator : [custom_data_generator.ipynb](../material/deep_learning/custom_data_generator.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/custom_data_generator.ipynb)

- [Keras Howto 모음](https://github.com/dhrim/keras_howto_2021)

<br>

# Data

## TensorFlow Dataset TFDS

- Dataset : [tensorflow_data_tfds.ipynb](../material/deep_learning/tensorflow_data_tfds.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/tensorflow_data_tfds.ipynb)
- image : [tensorflow_data_image.ipynb](../material/deep_learning/tensorflow_data_image.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/tensorflow_data_image.ipynb)

<br>

## Keras Sequence data generator

- 곡물 순차 데이터 예측 : [financial_data_predict_commodity_price.ipynb](../material/deep_learning/financial_data_predict_commodity_price.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/financial_data_predict_commodity_price.ipynb) 

<br>

# DNN, CNN

- 분류기로서의 DNN
    - 속성 데이터 IRIS 분류 실습 : [dnn_iris_classification.ipynb](../material/deep_learning/dnn_iris_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/dnn_iris_classification.ipynb)

- 영상 데이터의 이해 : [deep_learning_intro.pptx](../material/deep_learning/deep_learning_intro.pptx)

- 영상 분류기로서의 DNN
    - 흑백 영상 데이터 MNIST 분류 실습 : [dnn_mnist.ipynb](../material/deep_learning/dnn_mnist.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/dnn_mnist.ipynb)
    - 흑백 영상 fashion MNIST 분류 : [dnn_fashion_mnist.ipynb](../material/deep_learning/dnn_fashion_mnist.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/dnn_fashion_mnist.ipynb)

- 영상 분류기로서의 CNN
    - CNN의 이해 : [deep_learning_intro.pptx](../material/deep_learning/deep_learning_intro.pptx)
    - 흑백 영상 데이터 MNIST 영상분류 : [cnn_mnist.ipynb](../material/deep_learning/cnn_mnist.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/cnn_mnist.ipynb)
    - CIFAR10 컬러영상분류 : [cnn_cifar10.ipynb](../material/deep_learning/cnn_cifar10.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/cnn_cifar10.ipynb)


<br>

# 전이학습

- 전이학습
    [VGG16_classification_and_cumtom_data_training.ipynb](../material/deep_learning/VGG16_classification_and_cumtom_data_training.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/VGG16_classification_and_cumtom_data_training.ipynb)
    - 커스텀 데이터 VGG 데이터 분류 실습 : [flowers.zip](./deep_learning/data/flowers.zip)
        - [practice_custom_image_classification.ipynb](../material/deep_learning/practice_custom_image_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/practice_custom_image_classification.ipynb)
        - [real_practice_glaucoma_classification.ipynb](../material/deep_learning/real_practice_glaucoma_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/real_practice_glaucoma_classification.ipynb)
    - 영상 데이터 예측 : [cat_with_glasses.ipynb](../material/deep_learning/cat_with_glasses.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/cat_with_glasses.ipynb)

- GradCAM : [grad_cam.ipynb](../material/deep_learning/grad_cam.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/grad_cam.ipynb)

<br>

# AutoEncoder

- Keras Functional API  : [functional_api.ipynb](../material/deep_learning/functional_api.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/functional_api.ipynb)

- AutoEncoder
    - AutoEncoder 실습 : [autoencoder.ipynb](../material/deep_learning/autoencoder.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/autoencoder.ipynb)
    - 디노이징 AutoEncoder : [denoising_autoencoder.ipynb](../material/deep_learning/denoising_autoencoder.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/denoising_autoencoder.ipynb)
    - Super Resolution : [mnist_super_resolution.ipynb](../material/deep_learning/mnist_super_resolution.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/mnist_super_resolution.ipynb)

<br>

# 영상 분할

- 영상 분할(Segementation)
    - U-Net을 사용한 영상 분할 : [unet_segementation.ipynb](../material/deep_learning/unet_segementation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/unet_segementation.ipynb)
        - U-Net을 사용한 영상 분할 실습 - 거리 영상 : [unet_setmentation_practice.ipynb](../material/deep_learning/unet_setmentation_practice.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/unet_setmentation_practice.ipynb)
        - U-Net을 사용한 영상 분할 실습 - MRI : [MRI_images.zip](https://github.com/dhrim/deep_learning_data/raw/master/MRI_images.zip)        
    - M-Net을 사용한 영상 분할 : [mnet_segementation.ipynb](../material/deep_learning/mnet_segementation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/mnet_segementation.ipynb)
    - U-Net을 사용한 컬러 영상 분할 : [unet_segementation_color_image.ipynb](../material/deep_learning/unet_segementation_color_image.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/unet_segementation_color_image.ipynb)
    - U-Net을 사용한 다중 레이블 분할 : [unet_segmentation_multi_label.ipynb](../material/deep_learning/unet_segmentation_multi_label.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/unet_segmentation_multi_label.ipynb)

<br>

# 기타 영상 데이터 작업

- 물체 탐지
   - 물체 탐지의 이해
   - YOLO 적용 방법 실습 : [object_detection.md](../material/deep_learning/object_detection.md)

- 얼굴 인식 : [20220113_face_recognition_with_2_models.ipynb](../material/deep_learning/20220113_face_recognition_with_2_models.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/20220113_face_recognition_with_2_models.ipynb)
- 포즈 추출 : [open_pose_using_template.ipynb](../material/deep_learning/open_pose_using_template.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/open_pose_using_template.ipynb)
- web cam + colab 실시간 포즈 추출 : [tf_pose_estimation_with_webcam.ipynb](../material/deep_learning/tf_pose_estimation_with_webcam.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/tf_pose_estimation_with_webcam.ipynb)


- 얼굴 인식
    - 얼굴 위치 탐지 실습 : [track_faces_on_video_realtime.ipynb](../material/deep_learning/track_faces_on_video_realtime.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/track_faces_on_video_realtime.ipynb)
    - 얼굴 감정 분류 실습 : [face_emotion_classification.ipynb](../material/deep_learning/face_emotion_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/face_emotion_classification.ipynb)
- 스타일 변환 : https://www.tensorflow.org/tutorials/generative/style_transfer?hl=ko


<br>

# RNN

- RNN
    - text 데이터의 이해 : [deep_learning_intro.pptx](../material/deep_learning/deep_learning_intro.pptx)
    - RNN의 이해 : [deep_learning_intro.pptx](../material/deep_learning//deep_learning_intro.pptx)
    - RNN을 사용한 영화 평가 데이터 IMDB 분류 : [rnn_text_classification.ipynb](../material/deep_learning/rnn_text_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/rnn_text_classification.ipynb)
    - RNN을 사용한 다음 문자 생성 : [rnn_next_character_prediction.ipynb](../material/deep_learning/rnn_next_character_prediction.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/rnn_next_character_prediction.ipynb)
    - RNN을 사용한 덧셈 결과 생성 : [seq2seq_addition_using_rnn.ipynb](../material/deep_learning/seq2seq_addition_using_rnn.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/seq2seq_addition_using_rnn.ipynb)
    - RNN을 사용한 덧셈 결과 분류 : [rnn_addition_text_classication.ipynb](../material/deep_learning/rnn_addition_text_classication.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/rnn_addition_text_classication.ipynb)     
    - Bert를 사용한 다음 단어 예측 : [next_word_prediction.ipynb](../material/deep_learning/next_word_prediction.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/next_word_prediction.ipynb)

<br>

# 시계열 데이터

- 시계열 데이터 처리 : [treating_sequence_data.ipynb](../material/deep_learning/treating_sequence_data.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/treating_sequence_data.ipynb)
- 시계열 데이터 예측 : [weather_forecasting.ipynb](../material/deep_learning/weather_forecasting.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/weather_forecasting.ipynb)
- 시계열 데이터 분류 : [real_practice_classify_semiconductor_time_series_data.ipynb](../material/deep_learning/real_practice_classify_semiconductor_time_series_data.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/real_practice_classify_semiconductor_time_series_data.ipynb)


<br>

# 자연어 처리

- 한글 NLP    
    - RNN을 사용한 한글 영화 평가 데이터 분류 : [korean_word_sequence_classification.ipynb](../material/deep_learning/korean_word_sequence_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/korean_word_sequence_classification.ipynb)
    - Bert를 사용한 한글 영화 평가 데이터 분류 : [korean_word_sequence_classification_with_bert.ipynb](../material/deep_learning/korean_word_sequence_classification_with_bert.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/korean_word_sequence_classification_with_bert.ipynb)
    - Bert를 사용한 한글 문장 간 관계 분류 : [korean_sentence_relation_classification_with_bert.ipynb](../material/deep_learning/korean_sentence_relation_classification_with_bert.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/korean_sentence_relation_classification_with_bert.ipynb)
    - Bert를 사용한 한글 문장 간 관계값 예측 : [korean_sentence_relation_regression_with_bert.ipynb](../material/deep_learning/korean_sentence_relation_regression_with_bert.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/korean_sentence_relation_regression_with_bert.ipynb)
    - Bert를 사용한 한글 문장 간 관계 분류, 커스텀 vocab : [korean_sentence_relation_classification_with_bert_with_custom_vocab.ipynb](../material/deep_learning/korean_sentence_relation_classification_with_bert_with_custom_vocab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/korean_sentence_relation_classification_with_bert_with_custom_vocab.ipynb)
    - Bert를 사용한 괄호 단어 예측 : [korean_mask_completion_with_bert.ipynb](../material/deep_learning/korean_mask_completion_with_bert.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/korean_mask_completion_with_bert.ipynb) 


<br>

# GAN

- GAN
    - GAN의 이해 : [deep_learning_intro.pptx](../material/deep_learning//deep_learning_intro.pptx), 
        - 이상탐지 관련 GAN 설명 : [deep_learning_anomaly_detection.pptx](../material/deep_learning/deep_learning_anomaly_detection.pptx)
    - GAN을 사용한 MNIST 학습 실습 : [wgan_gp_mnist.ipynb](../material/deep_learning/wgan_gp_mnist.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/wgan_gp_mnist.ipynb)
    - GAN을 사용한 fashion MNIST 학습 실습 : [wgan_gp_fashion_mnnist.ipynb](../material/deep_learning/wgan_gp_fashion_mnnist.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/wgan_gp_fashion_mnnist.ipynb)
    - GAN을 사용한 CIFAR10 학습 실습 : [wgan_gp_cifar10.ipynb](../material/deep_learning/wgan_gp_cifar10.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/wgan_gp_cifar10.ipynb)
    - Conditional GAN의 이해 : [anomaly_detection_using_gan.pptx](../material/deep_learning/anomaly_detection_using_gan.pptx)
    - Cycle GAN의 이해 : [cycle_gan.pdf](../material/deep_learning/cycle_gan.pdf)


<br>

# 강화 학습

- 강화학습 이해하기 : [deep_learning_intro.pptx](../material/deep_learning//deep_learning_intro.pptx)


<br>


# 금융 데이터

- 시계열 데이터
    - 곡물 데이터 예측 : [financial_data_predict_commodity_price.ipynb](../material/deep_learning/financial_data_predict_commodity_price.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/financial_data_predict_commodity_price.ipynb) 
    - 구글 주가 분류 : [financial_data_classify_stock_price.ipynb](../material/deep_learning/financial_data_classify_stock_price.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/financial_data_classify_stock_price.ipynb) 

- 이상 탐지
    - 카드 사기 탐지 : [financial_data_detect_fraud_card.ipynb](../material/deep_learning/financial_data_detect_fraud_card.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/financial_data_detect_fraud_card.ipynb) 
    - fashion MNIST 이상 탐지 : [anomaly_detection_fahsion_mnist.ipynb](../material/deep_learning/anomaly_detection_fahsion_mnist.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/anomaly_detection_fahsion_mnist.ipynb) 

- 속성 데이터
    집값 예측 : [financial_data_classify_house_price.ipynb](../material/deep_learning/financial_data_classify_house_price.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/financial_data_classify_house_price.ipynb) 

<br>

# 추천

- 추천 시스템 소개 : [recommendation.ipynb](../material/deep_learning/recommendation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/recommendation.ipynb) 
- 실습 데이터
    - https://www.kaggle.com/c/instacart-market-basket-analysis/data
    - https://www.kaggle.com/niharika41298/netflix-visualizations-recommendation-eda/data?select=to_read.csv

- 영상 검색
    - 영상 검색 - by CNN : [image_search_by_CNN.ipynb](../material/deep_learning/image_search_by_CNN.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/image_search_by_CNN.ipynb)
    - 영상 검색 - by Conv AutoEncoder : [image_search_by_ConvAutoEncoder.ipynb](../material/deep_learning/image_search_by_ConvAutoEncoder.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/image_search_by_ConvAutoEncoder.ipynb)    
    - 영상 검색 - by 샴 네크웤 : [image_search_by_siamese_network.ipynb](../material/deep_learning/image_search_by_siamese_network.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/image_search_by_siamese_network.ipynb)
    - 실습 데이터 : BMW car data : http://ai.stanford.edu/~jkrause/car196/bmw10_release.tgz (10개 폴더에 이미지들)

- 소리 검색
    - 소리 검색 - urban sound : [sound_search_urban_sound.ipynb](../material/deep_learning/sound_search_urban_sound.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/sound_search_urban_sound.ipynb)
    - 소리 검색 - enviroment sound : [sound_search_urban_sound.ipynb](../material/deep_learning/sound_search_urban_sound.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/sound_search_urban_sound.ipynb)



<br>

# Template

- 속성 데이터
    - 예측 : [template_attribute_data_regression.ipynb](../material/deep_learning/template_attribute_data_regression.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/template_attribute_data_regression.ipynb)
    - 분류 : [template_attribute_data_classification.ipynb](../material/deep_learning/template_attribute_data_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/template_attribute_data_classification.ipynb)
    - 2진 분류 : [template_attribute_data_binary_classification.ipynb](../material/deep_learning/template_attribute_data_binary_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/template_attribute_data_binary_classification.ipynb)    

- 영상 데이터
    - 예측 - vanilla CNN : [template_image_data_vanilla_cnn_regression.ipynb](../material/deep_learning/template_image_data_vanilla_cnn_regression.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/template_image_data_vanilla_cnn_regression.ipynb)
    - 예측 - 전이학습 : [template_image_data_transfer_learning_regression.ipynb](../material/deep_learning/template_image_data_transfer_learning_regression.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/template_image_data_transfer_learning_regression.ipynb)
    - 분류 - vanilla CNN : [template_image_data_vanilla_cnn_classification.ipynb](../material/deep_learning/template_image_data_vanilla_cnn_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/template_image_data_vanilla_cnn_classification.ipynb)
    - 분류 - 전이학습 : [template_image_data_transfer_learning_classification.ipynb](../material/deep_learning/template_image_data_transfer_learning_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/template_image_data_transfer_learning_classification.ipynb)
    - 2진 분류 - vanilla CNN : [template_image_data_vanilla_cnn_binary_classification.ipynb](../material/deep_learning/template_image_data_vanilla_cnn_binary_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/template_image_data_vanilla_cnn_binary_classification.ipynb)
    - 2진 분류 - 전이학습 : [template_image_data_transfer_learning_binary_classification.ipynb](../material/deep_learning/template_image_data_transfer_learning_binary_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/template_image_data_transfer_learning_binary_classification.ipynb)

- 순차열 데이터
    - 숫자열
        - 단일 숫자열 예측 : [template_numeric_sequence_data_prediction.ipynb](../material/deep_learning/template_numeric_sequence_data_prediction.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/template_numeric_sequence_data_prediction.ipynb)
        - 단일 숫자열 분류 : [template_numeric_sequence_data_classification.ipynb](../material/deep_learning/template_numeric_sequence_data_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/template_numeric_sequence_data_classification.ipynb)
        - 다중 숫자열 분류 : [template_multi_numeric_sequence_data_classification.ipynb](../material/deep_learning/template_multi_numeric_sequence_data_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/template_multi_numeric_sequence_data_classification.ipynb) 
        - 다중 숫자열 다중 예측 : [template_multi_numeric_sequence_data_multi_prediction.ipynb](../material/deep_learning/template_multi_numeric_sequence_data_multi_prediction.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/template_multi_numeric_sequence_data_multi_prediction.ipynb)
        - 다중 숫자열 단일 예측 : [template_multi_numeric_sequence_data_one_prediction.ipynb](../material/deep_learning/template_multi_numeric_sequence_data_one_prediction.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/template_multi_numeric_sequence_data_one_prediction.ipynb)
        - sequence DataGenerator : [weather_forecasting.ipynb](../material/deep_learning/weather_forecasting.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/weather_forecasting.ipynb)        
    - 문자열
        - 문자열 예측 : [template_text_sequence_data_prediction.ipynb](../material/deep_learning/template_text_sequence_data_prediction.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/template_text_sequence_data_prediction.ipynb)
        - 문자열 분류 : [template_text_sequence_data_classification.ipynb](../material/deep_learning/template_text_sequence_data_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/template_text_sequence_data_classification.ipynb)
        - 문자열 연속 예측 : [template_text_data_sequential_generation.ipynb](../material/deep_learning/template_text_data_sequential_generation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/template_text_data_sequential_generation.ipynb)
    - 단어열
        - 단어열 분류 : [template_word_sequence_data_classification.ipynb](../material/deep_learning/template_word_sequence_data_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/template_word_sequence_data_classification.ipynb)
        - 단어열 예측 : [template_word_sequence_data_prediction.ipynb](../material/deep_learning/template_word_sequence_data_prediction.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/template_word_sequence_data_prediction.ipynb)
        - 한글 단어열 분류 : [template_korean_word_sequence_data_classification.ipynb](../material/deep_learning/template_korean_word_sequence_data_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/template_korean_word_sequence_data_classification.ipynb)
        - Bert를 사용한 한글 문장 간 관계 분류 : [korean_sentence_relation_classification_with_bert.ipynb](../material/deep_learning/korean_sentence_relation_classification_with_bert.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/korean_sentence_relation_classification_with_bert.ipynb)
        - Bert를 사용한 한글 문장 간 관계값 예측 : [korean_sentence_relation_regression_with_bert.ipynb](../material/deep_learning/korean_sentence_relation_regression_with_bert.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/korean_sentence_relation_regression_with_bert.ipynb)

- 추천
    - TensorFlow Recommendations 템플릿 : [TFRS_recommendation_template.ipynb](../material/deep_learning/TFRS_recommendation_template.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/TFRS_recommendation_template.ipynb)
 


<br>

# 기타 howto
- multi-label classification : [multi_label_classificaiton.ipynb](../material/deep_learning/multi_label_classificaiton.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/multi_label_classificaiton.ipynb)
- ROC, AUC, Confusion Matrix 그리기 : [roc_auc_confusion_matric.ipynb](../material/deep_learning/roc_auc_confusion_matric.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/roc_auc_confusion_matric.ipynb)
- cross validation : [cross_validation.ipynb](../material/deep_learning/cross_validation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/cross_validation.ipynb)
- 앙상블 : [ensemble.ipynb](../material/deep_learning/ensemble.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/ensemble.ipynb) 
- ImageDataGenerator를 사용한 데이터 증강 : [data_augmentation_using_ImageDadtaGenerator.ipynb](../material/deep_learning/data_augmentation_using_ImageDadtaGenerator.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/data_augmentation_using_ImageDadtaGenerator.ipynb) 
- PCA, T-SNE : [PCA_TSNE.ipynb](../material/deep_learning/PCA_TSNE.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/PCA_TSNE.ipynb) 
- Colab에서 TensorBoard 사용 : [tensorboard_in_colab.ipynb](../material/deep_learning/tensorboard_in_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/tensorboard_in_colab.ipynb) 
- 이미지 crop과 resize : [image_crop_and_resize.ipynb](../material/deep_learning/image_crop_and_resize.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/image_crop_and_resize.ipynb) 

<br>

# 성능 개선

- 성능 개선 개요 : [deep_learning_intro.pptx](../material/deep_learning/deep_learning_intro.pptx)
- 오버피팅 처리 : [dnn_in_keras_overfitting.ipynb](../material/deep_learning/dnn_in_keras_overfitting.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/dnn_in_keras_overfitting.ipynb)
- 데이터 수와 성능 : [data_count_and_overfitting.ipynb](../material/deep_learning/data_count_and_overfitting.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/data_count_and_overfitting.ipynb)
- weight 초기화와 성능 : [dnn_in_keras_weight_init.ipynb](../material/deep_learning/dnn_in_keras_weight_init.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/dnn_in_keras_weight_init.ipynb)
- normalization과 성능 : [normalization_and_performance.ipynb](../material/deep_learning/normalization_and_performance.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/normalization_and_performance.ipynb)
- 불균등 데이터 처리 : [treating_imbalanced_data.ipynb](../material/deep_learning/treating_imbalanced_data.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/treating_imbalanced_data.ipynb)
- IMDB 분류에 적용 : [treating_overfitting_with_imdb.ipynb](../material/deep_learning/treating_overfitting_with_imdb.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/treating_overfitting_with_imdb.ipynb)
- MNIST CNN에 callback과 오버피팅 처리 적용 : [boston_house_price_regression.ipynb](../material/deep_learning/boston_house_price_regression.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/boston_house_price_regression.ipynb)


<br>

# 기타

- 알파고 이해하기 : [understanding_ahphago.pptx](../material/deep_learning/understanding_ahphago.pptx)

<br>

# 기타 실습

- 영상 데이터 분류
    - 화재 영상 분류 : [fire_scene_classification.ipynb](../material/deep_learning/fire_scene_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/fire_scene_classification.ipynb)    
    - wafer map 영상 분류 : [real_practice_classify_semiconductor_wafermap.ipynb](../material/deep_learning/real_practice_classify_semiconductor_wafermap.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/real_practice_classify_semiconductor_wafermap.ipynb)
    - 엔진 블레이드 영상 분류 : [engine_blade_classification.ipynb](../material/deep_learning/engine_blade_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/engine_blade_classification.ipynb)


- 속성 데이터 분류
    - 심리설문 데이터 분류 : [real_practice_psychologial_test_classification.ipynb](../material/deep_learning/real_practice_psychologial_test_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/real_practice_psychologial_test_classification.ipynb)

- 소리 분류 : [classify_audio.ipynb](../material/deep_learning/classify_audio.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/classify_audio.ipynb)


<br>



<br>

# 참고 데이터

- Dacon 데이터
    - 글자에 숨겨진 MNIST 영상 분류 : [classification_hidden_mnist_in_lettern.ipynb](../material/deep_learning/classification_hidden_mnist_in_lettern.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/classification_hidden_mnist_in_lettern.ipynb)
    - 와인 속성 데이타 품질 분류 : [classification_wine_quality.ipynb](../material/deep_learning/classification_wine_quality.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/classification_wine_quality.ipynb)
    - 식술 성장 기간 예측 : [predict_plant_growing_interval.ipynb](../material/deep_learning/predict_plant_growing_interval.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/predict_plant_growing_interval.ipynb) 
- Kaggle 데이터
    - Kaggle x-ray 폐렴 분류 : https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
    - Kaggle 고양이 강아지 분류 : https://www.kaggle.com/tongpython/cat-and-dog


<br>


<br>

## 기존 오래된

- 윈도우 환경에서 linux command HowTo : [how_to_linux_command_on_windows.md](../material/env/how_to_linux_command_on_windows.md)
- Ubuntu 서버 설치하기(다소 오래된) : [2019-10-17_setup_server.pdf](../material/env/2019-10-17_setup_server.pdf)
- GCP에 VM생성하고 Colab 연결하기 : [GCP_VM_and_Colab.pdf](../material/env/GCP_VM_and_Colab.pdf)


<br>

## 기타 자료

- ML Classifiers : [ML_classifiers.ipynb](../material/deep_learning/ML_classifiers.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/ML_classifiers.ipynb)
- DNN regression. boston 집값 예측 : [boston_house_price_regression.ipynb](../material/deep_learning/boston_house_price_regression.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrim/DMC_2022/blob/master/material/deep_learning/boston_house_price_regression.ipynb) 
- [의학논문 리뷰](https://docs.google.com/presentation/d/1SZ-m4XVepS94jzXDL8VFMN2dh9s6jaN5fVsNhQ1qwEU/edit)
- GCP에 VM 생성하고 Colab 연결하기 : [create_GCP_VM.pdf](../material/deep_learning/create_GCP_VM.pdf)
- 흥미로운 딥러닝 결과 : [some_interesting_deep_learning.pptx](../material/deep_learning/some_interesting_deep_learning.pptx)
- yolo를 사용한 실시간 불량품 탐지 : https://drive.google.com/file/d/194UpsjG7MyEvWlmJeqfcocD-h-zy_4mR/view?usp=sharing
- YOLO를 사용한 자동차 번호판 탐지 : https://drive.google.com/file/d/1jlKzCaKj5rGRXIhwMXtYtVnx_XLauFiL/view?usp=sharing
- 딥러닝 이상탐지 : [deep_learning_anomaly_detection.pptx](../material/deep_learning/deep_learning_anomaly_detection.pptx)
- GAN을 사용한 생산설비 이상 탐지 : [anomaly_detection_using_gan.pptx](../material/deep_learning/anomaly_detection_using_gan.pptx)
- 이상탐지 동영상 : [drillai_anomaly_detect.mp4](../material/deep_learning/drillai_anomaly_detect.mp4)
- 훌륭한 논문 리스트 : https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap
- online CNN 시각화 자료 : https://poloclub.github.io/cnn-explainer/
- 서버 설치 기록 : [2019-10-17_setup_server.pdf](../material/env/2019-10-17_setup_server.pdf)
- GCP에 VM 생성하고 Colab 연결 : [GCP_VM_and_Colab.pdf](../material/env/GCP_VM_and_Colab.pdf)


<br>

# 교육에 사용된 외부 자료

- boston dynamics 1 : https://www.youtube.com/watch?v=_sBBaNYex3E
- boston dynamics 2 : https://www.youtube.com/watch?v=94nnAOZRg8k
- cart pole : https://www.youtube.com/watch?v=XiigTGKZfks
- bidirectional RNN : https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66
- alphago architecture : https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0
- u-net architecture : https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/
- upsampling : https://kharshit.github.io/blog/2019/02/15/autoencoder-downsampling-and-upsampling
- Denseness architecture : https://hoya012.github.io/blog/DenseNet-Tutorial-1/
- K-fold cross validation : https://m.blog.naver.com/PostView.nhn?blogId=dnjswns2280&logNo=221532535858&proxyReferer=https:%2F%2Fwww.google.com%2F
- M-net architecture : https://hzfu.github.io/proj_glaucoma_fundus.html  
- yolo 적용 예 블로그 : https://nero.devstory.co.kr/post/pj-too-real-03/
- GAN 위조 지폐 : http://mrkim.cloudy.so/board_KBEq62/175378
- GAN paper : https://arxiv.org/pdf/1406.2661.pdf
- Gan paper count : https://blog.naver.com/PostView.nhn?blogId=laonple&logNo=221201915691
- Conditional gan face generation example. https://github.com/Guim3/IcGAN
- Pinpointing example : https://www.geeks3d.com/20180425/nvidia-deep-learning-based-image-inpainting-demo-is-impressive/
- 동영상 스타일 변환 : https://www.youtube.com/watch?v=Khuj4ASldmU
- 얼굴 감정 인식 예 : http://www.astronomer.rocks/news/articleView.html?idxno=86084
- Papers with code : https://paperswithcode.com/
  

<br>


