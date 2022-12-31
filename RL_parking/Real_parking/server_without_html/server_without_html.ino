/*********
  https://randomnerdtutorials.com/esp32-esp8266-input-data-html-form/
  https://github.com/me-no-dev/ESPAsyncWebServer
*********/

#include "WiFi.h"
#include "esp_timer.h"
#include "Arduino.h"
#include "soc/soc.h"           // Disable brownout problems
#include "soc/rtc_cntl_reg.h"  // Disable brownout problems
#include "driver/rtc_io.h"
#include "driver/mcpwm.h"
#include <ESPAsyncWebServer.h>
#include <StringArray.h>
#include <FS.h>
#include <stdio.h>
#include "MPU6050_tockn.h"
#include <Wire.h>

esp_err_t esp_err;
const int buzzer = 25;
//pwm
const int f = 1000; //频率
const int c = 0;  //通道
const int r = 8;  //分辨率
const int d = 128; //占空比
float servo_duty_cycle_center = 7.5;
float servo_duty_cycle_differ_45 = 5;
float servo_duty_cycle_differ_30 = 2.5;

void move_forward();
void move_backward();
void motor_stop();
void turn_left_30();
void turn_right_30();
void turn_left_45();
void turn_right_45();
void straight();
void ring();
void stop_ring();


// LED pin
const int left_turn_led_pin = 32;
const int right_turn_led_pin = 33;

// encoder pin
const int encoder_pin = 2;

// motor pwm pin
const int motor_pwm_pin_A = 27;
const int motor_pwm_pin_B = 26;
// servo pwm pin
const int servo_pwm_pin = 13;

// Set your access point network credentials
const char* ssid = "xspnbxspnb";
const char* password = "xspnbxspnb";

// Create AsyncWebServer object on port 80
AsyncWebServer server(80);

// mpu6050
const int mpu_int = 5;
MPU6050 mpu6050(Wire);

// motor parameters
int motor_duty_cycle = 30;
int servo_turn_angle = 45;
float servo_duty_cycle_differ = 1.5;
float servo_duty_cycle;

// speed
int count = 0;
float car_speed = 0.0;
void IRAM_ATTR count_add() {
  count += 1;
}

// angle
float z_angle = 0.0;

// local time
unsigned long old_millis;

// function statement
void left_light_control(bool state);
void right_light_control(bool state);
void control_all_light(bool);
void motor_control(int motor_speed);
void servo_control(int servo_angle);
// void mpu6050_reinit();

//buzzer_pwm
const int buzzer_pwm_pin = 25;
const int buzzer_pwm_freq = 1000;
const int buzzer_pwm_channel = 7;
const int buzzer_pwm_resolution = 8;

void setup() {
  Serial.begin(115200);

  ledcSetup(c,f,r);
  ledcAttachPin(buzzer,c);
  
  mcpwm_gpio_init(MCPWM_UNIT_0, MCPWM0A, 26);
  mcpwm_gpio_init(MCPWM_UNIT_0, MCPWM0B, 27);
  mcpwm_gpio_init(MCPWM_UNIT_1, MCPWM1A, 15);
  mcpwm_config_t motor_pwm_config = {
    .frequency = 1000,
    .cmpr_a = 0,
    .cmpr_b = 0,
    .duty_mode = MCPWM_DUTY_MODE_0,
    .counter_mode = MCPWM_UP_COUNTER,
  };
  esp_err = mcpwm_init(MCPWM_UNIT_0, MCPWM_TIMER_0, &motor_pwm_config);
  Serial.println(esp_err);
  mcpwm_config_t servo_pwm_config;
  servo_pwm_config.frequency = 50;
  servo_pwm_config.cmpr_a = 0;
  servo_pwm_config.duty_mode = MCPWM_DUTY_MODE_0;
  servo_pwm_config.counter_mode = MCPWM_UP_COUNTER;
    
  esp_err = mcpwm_init(MCPWM_UNIT_1, MCPWM_TIMER_1, &servo_pwm_config);
  if (esp_err == 0)
    Serial.println("Setting motor pwm success!");
  else {
    Serial.print("Setting motor pwm fail, error code: ");
    Serial.println(esp_err);
  }
  mcpwm_start(MCPWM_UNIT_1, MCPWM_TIMER_1);
  // Serial port for debugging purposes
  ledcSetup(buzzer_pwm_channel, buzzer_pwm_freq, buzzer_pwm_resolution);
  ledcAttachPin(buzzer_pwm_pin, buzzer_pwm_channel);

  WiFi.mode(WIFI_AP);
  if(!WiFi.softAPConfig(IPAddress(192, 168, 4, 1), IPAddress(192, 168, 4, 1), IPAddress(255, 255, 0, 0))){
      Serial.println("AP Config Failed");
  }
  WiFi.softAP(ssid, password);

  IPAddress IP = WiFi.softAPIP();
  Serial.print("AP IP address: ");
  Serial.println(IP);

  // Turn-off the 'brownout detector'
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  // set led pinmode
  pinMode(left_turn_led_pin, OUTPUT);
  pinMode(right_turn_led_pin, OUTPUT);

  // encoder interrupt
  pinMode(encoder_pin, INPUT);
  attachInterrupt(encoder_pin, count_add, RISING);

  // motor pwm config
  mcpwm_gpio_init(MCPWM_UNIT_0, MCPWM0A, motor_pwm_pin_A);
  mcpwm_gpio_init(MCPWM_UNIT_0, MCPWM0B, motor_pwm_pin_B);
  esp_err = mcpwm_init(MCPWM_UNIT_0, MCPWM_TIMER_0, &motor_pwm_config);
  if (esp_err == 0)
    Serial.println("Setting motor pwm success!");
  else {
    Serial.print("Setting motor pwm fail, error code: ");
    Serial.println(esp_err);
  }

  // servo pwm config
  mcpwm_gpio_init(MCPWM_UNIT_1, MCPWM1A, servo_pwm_pin);
  servo_pwm_config.frequency = 50;
  servo_pwm_config.cmpr_a = 0;
  servo_pwm_config.duty_mode = MCPWM_DUTY_MODE_0;
  servo_pwm_config.counter_mode = MCPWM_UP_COUNTER;
  esp_err = mcpwm_init(MCPWM_UNIT_1, MCPWM_TIMER_1, &servo_pwm_config);
  if (esp_err == 0)
    Serial.println("Setting servo pwm success!");
  else {
    Serial.print("Setting servo pwm fail, error code: ");
    Serial.println(esp_err);
  }
  mcpwm_start(MCPWM_UNIT_1, MCPWM_TIMER_1);

/*
  Web Serve
*/

  // 准确控制
  server.on("/forward", HTTP_POST, [](AsyncWebServerRequest * request) {
    move_forward();
    request->send(200);
  });
  server.on("/backward", HTTP_POST, [](AsyncWebServerRequest * request) {
    move_backward();
    request->send(200);
  });

  server.on("/left", HTTP_POST, [](AsyncWebServerRequest * request) {
    turn_left_45();
    request->send(200);
  });
  
  server.on("/right", HTTP_POST, [](AsyncWebServerRequest * request) {
    turn_right_45();
    request->send(200);
  });

  server.on("/straight", HTTP_POST, [](AsyncWebServerRequest * request) {
    straight();
    request->send(200);
  });

  server.on("/stop", HTTP_POST, [](AsyncWebServerRequest * request) {
    motor_stop();
    request->send(200);
  });
  
  // Start server
  server.begin();
  
  pinMode(mpu_int, OUTPUT);
  digitalWrite(mpu_int, HIGH);
  Wire.begin(16, 17);
  mpu6050.begin();
  mpu6050.calcGyroOffsets(true);

  // set initial time
  old_millis = millis();
}

void loop() {
  mpu6050.update();
  z_angle = mpu6050.getAngleZ();

  delay(50);

  if(millis() - old_millis >= 500){
    car_speed = count / 18.0 / 21 * 6.2 * 3.14 * 1000 / 500;
    count = 0;
    old_millis = millis();
    
  }
}

// some functions

void move_backward() {
  Serial.println("--- move backward...");
  mcpwm_stop(MCPWM_UNIT_0, MCPWM_TIMER_0);
  mcpwm_set_duty(MCPWM_UNIT_0, MCPWM_TIMER_0, MCPWM_OPR_A, 0);
  mcpwm_set_duty(MCPWM_UNIT_0, MCPWM_TIMER_0, MCPWM_OPR_B, 70);
  mcpwm_start(MCPWM_UNIT_0, MCPWM_TIMER_0);
}

void move_forward() {
  mcpwm_stop(MCPWM_UNIT_0, MCPWM_TIMER_0);
  mcpwm_set_duty(MCPWM_UNIT_0, MCPWM_TIMER_0, MCPWM_OPR_A, 70);
  mcpwm_set_duty(MCPWM_UNIT_0, MCPWM_TIMER_0, MCPWM_OPR_B, 0);
  mcpwm_start(MCPWM_UNIT_0, MCPWM_TIMER_0);
  Serial.println("--- move forward...");
}

void motor_stop() {
  Serial.println("--- motor stop...");
  mcpwm_stop(MCPWM_UNIT_0, MCPWM_TIMER_0);
  mcpwm_set_duty(MCPWM_UNIT_0, MCPWM_TIMER_0, MCPWM_OPR_A, 100);
  mcpwm_set_duty(MCPWM_UNIT_0, MCPWM_TIMER_0, MCPWM_OPR_B, 100);
  mcpwm_start(MCPWM_UNIT_0, MCPWM_TIMER_0);
}

void turn_left_30() {
  mcpwm_set_duty(MCPWM_UNIT_1, MCPWM_TIMER_1, MCPWM_OPR_A, servo_duty_cycle_center - servo_duty_cycle_differ_30);
}

void turn_right_30() {
  mcpwm_set_duty(MCPWM_UNIT_1, MCPWM_TIMER_1, MCPWM_OPR_A, servo_duty_cycle_center + servo_duty_cycle_differ_30);
}

void turn_left_45() {
  mcpwm_set_duty(MCPWM_UNIT_1, MCPWM_TIMER_1, MCPWM_OPR_A, servo_duty_cycle_center - servo_duty_cycle_differ_45);
}

void turn_right_45() {
  mcpwm_set_duty(MCPWM_UNIT_1, MCPWM_TIMER_1, MCPWM_OPR_A, servo_duty_cycle_center + servo_duty_cycle_differ_45);
}

void straight() {
  mcpwm_set_duty(MCPWM_UNIT_1, MCPWM_TIMER_1, MCPWM_OPR_A, servo_duty_cycle_center);
}
