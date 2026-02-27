import { useCallback, useEffect, useState } from 'react';
import { Anchor, Box, Button, Divider, Image, Menu, Modal, Select, Slider, Stack, Text, Tooltip } from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import { IconChevronDown, IconRefresh } from '@tabler/icons-react';
import { MJSWAN_VERSION } from '../Version';
import FloatingPanel from './FloatingPanel';
import { LabeledInput } from './LabeledInput';
import { CommandSection } from './CommandSection';
import { SplatSection } from './SplatSection';
import {
  getCommandManager,
  type CommandDefinition,
  type SliderCommandConfig,
} from '../core/command';

export interface SelectOption {
  value: string;
  label: string;
}

interface ControlPanelProps {
  projects: SelectOption[];
  projectValue: string | null;
  projectLabel: string;
  onProjectChange: (value: string | null) => void;
  scenes: SelectOption[];
  sceneValue: string | null;
  onSceneChange: (value: string | null) => void;
  policies: SelectOption[];
  policyValue: string | null;
  onPolicyChange: (value: string | null) => void;
  /** Whether command controls are enabled */
  commandsEnabled?: boolean;
  /** Callback when reset button is pressed */
  onReset?: () => void;
  /** Splat loading */
  onLoadSplat?: (url: string, scale: number, groundOffset: number) => void;
  onClearSplat?: () => void;
  splatLoaded?: boolean;
}

/**
 * Format group name for display (e.g., "velocity" -> "Velocity")
 */
function formatGroupName(groupName: string): string {
  return groupName
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/**
 * SliderControl - Renders a slider for a slider command with horizontal layout
 */
function SliderControl({
  command,
  value,
  onChange,
  disabled,
}: {
  command: CommandDefinition;
  value: number;
  onChange: (id: string, value: number) => void;
  disabled?: boolean;
}) {
  const config = command.config as SliderCommandConfig;

  return (
    <Box
      pb="0.5em"
      px="xs"
      style={{
        display: 'flex',
        alignItems: 'center',
      }}
    >
      <Text
        c="dimmed"
        style={{
          fontSize: '0.875em',
          fontWeight: 450,
          lineHeight: '1.375em',
          letterSpacing: '-0.75px',
          width: '50%',
          flexShrink: 0,
        }}
      >
        {config.label}
      </Text>
      <Box style={{ width: '50%' }}>
        <Slider
          value={value}
          onChange={(val) => onChange(command.id, val)}
          min={config.min}
          max={config.max}
          step={config.step}
          size="xs"
          disabled={disabled}
          styles={{
            root: { padding: '0' },
            track: { height: 4 },
            thumb: { width: 12, height: 12 },
          }}
        />
      </Box>
    </Box>
  );
}

function ControlPanel(props: ControlPanelProps) {
  const {
    projects,
    projectValue,
    projectLabel,
    onProjectChange,
    scenes,
    sceneValue,
    onSceneChange,
    policies,
    policyValue,
    onPolicyChange,
    commandsEnabled = false,
    onReset,
    onLoadSplat,
    onClearSplat,
    splatLoaded = false,
  } = props;

  const [aboutModalOpened, { open: openAbout, close: closeAbout }] = useDisclosure(false);

  // Command state
  const [commands, setCommands] = useState<CommandDefinition[]>([]);
  const [commandGroups, setCommandGroups] = useState<string[]>([]);
  const [values, setValues] = useState<Record<string, number>>({});

  // Initialize commands from CommandManager
  useEffect(() => {
    const commandManager = getCommandManager();

    const updateCommands = () => {
      setCommands(commandManager.getCommands());
      setCommandGroups(commandManager.getCommandGroups());
      setValues(commandManager.getValues());
    };

    updateCommands();

    // Subscribe to command changes
    commandManager.addEventListener(updateCommands);

    return () => {
      commandManager.removeEventListener(updateCommands);
    };
  }, []);

  // Handle slider value changes
  const handleSliderChange = useCallback((id: string, value: number) => {
    const commandManager = getCommandManager();
    commandManager.setValue(id, value);
    setValues((prev) => ({ ...prev, [id]: value }));
  }, []);

  // Handle reset button click
  const handleReset = useCallback(() => {
    const commandManager = getCommandManager();
    commandManager.triggerButton('_system:reset');
    if (onReset) {
      onReset();
    }
  }, [onReset]);

  // Get slider commands for a specific group
  const getSliderCommandsForGroup = (groupName: string): CommandDefinition[] => {
    return commands.filter(
      (cmd) => cmd.groupName === groupName && cmd.config.type === 'slider'
    );
  };

  // Only show panel if we have data to display
  if (!projects.length && !scenes.length && !policies.length) {
    return null;
  }

  return (
    <>
    <Modal
      opened={aboutModalOpened}
      onClose={closeAbout}
      size="md"
      title={null}
      centered
      styles={{ body: { textAlign: 'center' } }}
    >
      <Stack gap="md" align="center">
        <Image src="./logo-color.svg" style={{ width: '8em', height: 'auto' }} />
        <Text size="xl" fw={700}>powered by mjswan</Text>
        <Text size="sm" c="dimmed">version {MJSWAN_VERSION}</Text>
        <Divider w="100%" />
        <Box pb="lg">
          <Anchor href="https://github.com/ttktjmt/mjswan" target="_blank" style={{ fontWeight: '600' }}>
            GitHub
          </Anchor>
          &nbsp;&nbsp;&bull;&nbsp;&nbsp;
          <Anchor href="https://mjswan.readthedocs.io" target="_blank" style={{ fontWeight: '600' }}>
            Documentation
          </Anchor>
        </Box>
      </Stack>
    </Modal>
    <FloatingPanel width="20em">
      <FloatingPanel.Handle>
        <Tooltip label={`mjswan ${MJSWAN_VERSION}`}>
          <Box
            component="a"
            onClick={(e) => { e.stopPropagation(); openAbout(); }}
            style={{ position: "absolute", cursor: "pointer", display: "flex", top: "0.8em", left: "0.9em" }}
          >
            <Image src="./logo.svg" style={{ width: "1.2em", height: "auto" }} />
          </Box>
        </Tooltip>
        <div style={{ width: "1.1em" }} />
        <FloatingPanel.HideWhenCollapsed>
          <Box
            px="xs"
            style={{
              flexGrow: 1,
              letterSpacing: "-0.5px",
              display: "flex",
              alignItems: "center",
              gap: "0.5em",
            }}
            pt="0.1em"
          >
            <span style={{ flexGrow: 1 }}>{projectLabel}</span>
            {projects.length > 1 && (
              <Menu position="bottom-start" offset={5}>
                <Menu.Target>
                  <Box
                    onClick={(e) => e.stopPropagation()}
                    style={{
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                    }}
                  >
                    <IconChevronDown size={16} />
                  </Box>
                </Menu.Target>
                <Menu.Dropdown onClick={(e) => e.stopPropagation()}>
                  {projects.map((project) => (
                    <Menu.Item
                      key={project.value}
                      onClick={(e) => {
                        e.stopPropagation();
                        onProjectChange(project.value);
                      }}
                      style={{
                        fontWeight: project.value === projectValue ? 600 : 400,
                        backgroundColor:
                          project.value === projectValue
                            ? "rgba(34, 139, 230, 0.1)"
                            : undefined,
                      }}
                    >
                      {project.label}
                    </Menu.Item>
                  ))}
                </Menu.Dropdown>
              </Menu>
            )}
          </Box>
        </FloatingPanel.HideWhenCollapsed>
        <FloatingPanel.HideWhenExpanded>
          <Box px="xs" style={{ flexGrow: 1, letterSpacing: "-0.5px" }} pt="0.1em">
            {projectLabel}
          </Box>
        </FloatingPanel.HideWhenExpanded>
      </FloatingPanel.Handle>
      <FloatingPanel.Contents>
        <Box pt="0.375em">
          {scenes.length > 0 && (
            <LabeledInput id="scene-select" label="Scene">
              <Select
                id="scene-select"
                placeholder="Select scene"
                data={scenes}
                value={sceneValue}
                onChange={onSceneChange}
                size="xs"
                radius="xs"
                searchable
                clearable={false}
                styles={{
                  input: { minHeight: '1.625rem', height: '1.625rem', padding: '0.5em' },
                }}
                comboboxProps={{ zIndex: 1000 }}
              />
            </LabeledInput>
          )}

          {policies.length > 0 && (
            <LabeledInput id="policy-select" label="Policy">
              <Select
                id="policy-select"
                placeholder="Select policy"
                data={policies}
                value={policyValue}
                onChange={onPolicyChange}
                size="xs"
                radius="xs"
                searchable
                clearable
                styles={{
                  input: { minHeight: '1.625rem', height: '1.625rem', padding: '0.5em' },
                }}
                comboboxProps={{ zIndex: 1000 }}
              />
            </LabeledInput>
          )}

          {/* Command Groups - only show if there are commands */}
          {commandGroups.length > 0 && commands.filter(cmd => cmd.config.type === 'slider').length > 0 && (
            <>
              {commandGroups.map((groupName) => {
                const groupCommands = getSliderCommandsForGroup(groupName);
                if (groupCommands.length === 0) return null;

                return (
                  <CommandSection
                    key={groupName}
                    label={formatGroupName(groupName)}
                    expandByDefault={true}
                  >
                    {groupCommands.map((command) => (
                      <SliderControl
                        key={command.id}
                        command={command}
                        value={values[command.id] ?? 0}
                        onChange={handleSliderChange}
                        disabled={!commandsEnabled}
                      />
                    ))}
                  </CommandSection>
                );
              })}
            </>
          )}

          <SplatSection onLoad={onLoadSplat} onClear={onClearSplat} loaded={splatLoaded} />

          {/* Reset Button - always at bottom */}
          <Divider my="xs" mx="xs" />
          <Box px="xs" pb="xs">
            <Button
              variant="light"
              color="red"
              size="xs"
              fullWidth
              leftSection={<IconRefresh size={14} />}
              onClick={handleReset}
            >
              Reset
            </Button>
          </Box>
        </Box>
      </FloatingPanel.Contents>
    </FloatingPanel>
    </>
  );
}

export default ControlPanel;
