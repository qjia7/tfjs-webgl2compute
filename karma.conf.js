const karmaTypescriptConfig = {
  tsconfig: 'tsconfig.json',
  // Disable coverage reports and instrumentation by default for tests
  coverageOptions: {instrumentation: false},
  reports: {}
};

module.exports = function(config) {
  config.set({
    basePath: '',
    frameworks: ['jasmine', 'karma-typescript'],
    files: [
      'src/*.ts'
    ],
    preprocessors: {'**/*.ts': ['karma-typescript']},
    karmaTypescriptConfig,
    reporters: ['progress', 'karma-typescript'],
    port: 9876,
    colors: true,

    // level of logging
    // possible values: config.LOG_DISABLE || config.LOG_ERROR || config.LOG_WARN || config.LOG_INFO || config.LOG_DEBUG
    logLevel: config.LOG_INFO,

    // enable / disable watching file and executing tests whenever any file changes
    autoWatch: false,

    browsers: ['Chrome'],

    singleRun: true
  })
}
