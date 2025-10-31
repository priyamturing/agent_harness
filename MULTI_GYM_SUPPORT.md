# Multi-Gym Support Implementation

## Overview

This document describes the multi-gym support feature that allows you to configure and run benchmarks against multiple MCP server environments (called "gyms").

## Features

- **Multiple Environment Support**: Configure local, staging, production, or any custom MCP server environments
- **Flexible Configuration**: Support for both simple and complex MCP server configurations
- **Custom Headers**: Add custom HTTP headers per gym (e.g., API keys, access tokens)
- **Auto-derived URLs**: SQL runner URLs are automatically derived from MCP URLs if not specified
- **Default Gym**: Set a default gym to use when none is specified
- **Easy Switching**: Switch between environments using the `--gym` flag

## Configuration

### Creating the Configuration File

1. Copy the example configuration:
   ```bash
   cp gyms.example.json gyms.json
   ```

2. Edit `gyms.json` to add your environments

### Configuration Format

```json
{
  "gyms": {
    "gym-name": {
      "name": "gym-name",
      "description": "Human-readable description",
      "url": "http://server:port/mcp",
      "transport": {
        "type": "streamable_http"
      },
      "sql_runner_url": "http://server:port/api/sql-runner",
      "headers": {
        "x-custom-header": "value"
      }
    }
  },
  "default": "gym-name"
}
```

### Configuration Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | No | Display name for the gym. Defaults to the key name. |
| `description` | No | Human-readable description of the environment. |
| `url` | Yes | The MCP server URL (e.g., `http://localhost:8015/mcp`). |
| `transport` | No | Transport configuration object with a `type` field. Defaults to `{"type": "streamable_http"}`. |
| `sql_runner_url` | No | URL for the SQL runner endpoint. Auto-derived from `url` if not provided. |
| `headers` | No | Custom HTTP headers to send with every request. `x-database-id` is automatically added per-run. |

### Example Configurations

#### Simple Local Setup
```json
{
  "gyms": {
    "local": {
      "url": "http://localhost:8015/mcp",
      "transport": {
        "type": "streamable_http"
      }
    }
  },
  "default": "local"
}
```

#### With Custom Headers
```json
{
  "gyms": {
    "teams-management": {
      "name": "teams-management",
      "description": "Teams management MCP server",
      "url": "http://localhost:8000/mcp",
      "headers": {
        "x-teams-access-token": "static_token_alex"
      },
      "transport": {
        "type": "http"
      }
    }
  },
  "default": "teams-management"
}
```

#### Multiple Environments
```json
{
  "gyms": {
    "local": {
      "description": "Local development",
      "url": "http://localhost:8015/mcp",
      "transport": {"type": "streamable_http"}
    },
    "staging": {
      "description": "Staging environment",
      "url": "http://staging.example.com:8015/mcp",
      "transport": {"type": "streamable_http"},
      "headers": {
        "x-api-key": "staging-key"
      }
    },
    "production": {
      "description": "Production environment",
      "url": "http://prod.example.com:8015/mcp",
      "transport": {"type": "streamable_http"},
      "headers": {
        "x-api-key": "prod-key"
      }
    }
  },
  "default": "local"
}
```

## Usage

### List Available Gyms

```bash
python -m jira_mcp_benchmark list-gyms
```

This displays a table of all configured gyms with their URLs and transport types.

### Run Against a Specific Gym

```bash
# Use the default gym
python -m jira_mcp_benchmark run --harness-file harness/test.json

# Specify a gym explicitly
python -m jira_mcp_benchmark run --harness-file harness/test.json --gym staging

# Use a custom config file
python -m jira_mcp_benchmark run --harness-file harness/test.json --gym production --gym-config /path/to/gyms.json
```

### Examples

```bash
# Run against local development server
python -m jira_mcp_benchmark run \
  --harness-file harness/test.json \
  --gym local \
  --model gpt-4o

# Run against staging with multiple models
python -m jira_mcp_benchmark run \
  --harness-file harness/test.json \
  --gym staging \
  --model gpt-4o \
  --model claude-sonnet-4-5 \
  --runs 3

# Run in background against production
python -m jira_mcp_benchmark run \
  --harness-file harness/test.json \
  --gym production \
  --background
```

## Implementation Details

### Files Added

1. **`src/jira_mcp_benchmark/gym_loader.py`**: Core gym configuration loader
   - `GymConfig`: Dataclass representing a gym configuration
   - `GymRegistry`: Registry for managing gym configurations
   - Helper functions for loading and accessing gym configurations

2. **`gyms.example.json`**: Example configuration file with multiple gym setups

### Files Modified

1. **`src/jira_mcp_benchmark/cli.py`**:
   - Added `--gym` and `--gym-config` options to the `run` command
   - Added `list-gyms` command
   - Updated `_execute_run()` to accept and use `gym_config`
   - Updated all run functions to pass gym configuration
   - Added gym information to run artifacts

2. **`README.md`**: Updated with gym configuration documentation

### Key Features

1. **Flexible Header Management**: Custom headers are merged with the per-run `x-database-id` header
2. **Backward Compatibility**: Supports both `url`/`mcp_url` field names and nested/flat transport configuration
3. **Smart Defaults**: SQL runner URLs are automatically derived if not specified
4. **Error Handling**: Clear error messages when gyms are not found or misconfigured
5. **Run Tracking**: Gym name and URL are stored in run artifacts for traceability

## Migration Guide

If you were previously using hardcoded MCP URLs, you can migrate by:

1. Creating a `gyms.json` file with your current configuration:
   ```json
   {
     "gyms": {
       "local": {
         "url": "http://localhost:8015/mcp",
         "transport": {"type": "streamable_http"}
       }
     },
     "default": "local"
   }
   ```

2. Your existing commands will continue to work, using the default gym

3. You can now add additional environments and switch between them using `--gym`

## Notes

- The `gyms.json` file is gitignored by default (via `*.json` in `.gitignore`) to prevent accidental exposure of sensitive URLs or API keys
- Always use `gyms.example.json` as a template for documentation
- The `x-database-id` header is always set per-run and cannot be overridden in gym configuration
- Run artifacts include the gym name and URL for traceability

