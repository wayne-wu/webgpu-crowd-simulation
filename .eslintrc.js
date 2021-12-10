module.exports = {
  parser: "@typescript-eslint/parser",
  extends: [
    "plugin:@typescript-eslint/recommended",
    "plugin:react/recommended",
    // "plugin:prettier/recommended",
    // "prettier/react",
    // "prettier/@typescript-eslint"
  ],
  plugins: [
    "@typescript-eslint",
    "react",
    // "prettier"
  ],
  rules: {
    "react/react-in-jsx-scope": "off",
    "react/prop-types": "off",
    "@typescript-eslint/no-unused-vars": "off",
    "@typescript-eslint/no-inferrable-types": "off",
    "no-var": "off",
    "no-const": "off",
    "prefer-const": "off"
  },
  globals: {
    React: "writable"
  },
};
